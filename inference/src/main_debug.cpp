//
// Debug walkthrough of the ternary inference engine.
// Runs ONE image, prints shapes/dtypes/values at every stage.
// Build: g++ -std=c++17 -O3 -march=native -mavx2 -mavxvnni -fopenmp -o build/debug src/main_debug.cpp -fopenmp -lm
//
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include "engine.h"

// ─── Helpers ────────────────────────────────────────────────

static void print_tensor_stats(const char* name, const Tensor& t) {
    int tot = t.total();
    float mn = *std::min_element(t.data.begin(), t.data.begin() + tot);
    float mx = *std::max_element(t.data.begin(), t.data.begin() + tot);
    double sum = 0; for (int i = 0; i < tot; i++) sum += t.data[i];
    printf("  %-30s  Tensor FP32  [%d,%d,%d,%d]  %d elements  %.1f KB\n"
           "                                  range=[%.4f, %.4f]  mean=%.4f\n",
           name, t.n, t.c, t.h, t.w, tot, tot*4.0/1024, mn, mx, sum/tot);
}

static void print_u8_stats(const char* name, const TensorU8& t) {
    int tot = t.total();
    uint8_t mn = *std::min_element(t.data.begin(), t.data.begin() + tot);
    uint8_t mx = *std::max_element(t.data.begin(), t.data.begin() + tot);
    double sum = 0; for (int i = 0; i < tot; i++) sum += t.data[i];
    int zeros = 0; for (int i = 0; i < tot; i++) if (t.data[i] == 0) zeros++;
    printf("  %-30s  TensorU8     [%d,%d,%d,%d]  %d elements  %.1f KB  scale=%.6f\n"
           "                                  range=[%d, %d]  mean=%.1f  zeros=%d (%.1f%%)\n"
           "                                  real_range=[%.4f, %.4f]\n",
           name, t.n, t.c, t.h, t.w, tot, tot/1024.0, t.scale,
           mn, mx, sum/tot, zeros, 100.0*zeros/tot,
           mn * t.scale, mx * t.scale);
}

static void print_kernel_info(const char* name, const FusedConv& layer, const I8KernelData& kd) {
    auto& p = layer.p;
    int noc = p.out_channels;
    int fan_in = kd.col_rows;

    // Count weight values
    int n_pos=0, n_neg=0, n_zero=0;
    if (layer.is_ternary) {
        for (int o = 0; o < noc; o++)
            for (int k = 0; k < fan_in; k++) {
                int8_t w = layer.weights_i8[o * fan_in + k];
                if (w > 0) n_pos++; else if (w < 0) n_neg++; else n_zero++;
            }
    }

    printf("  %-30s  %s  OC=%d IC=%d k=%dx%d s=%d p=%d\n",
           name, layer.is_ternary ? "TERNARY" : "FP32/INT8",
           noc, p.in_channels, p.kernel_h, p.kernel_w, p.stride, p.padding);
    printf("                                  col_rows=%d (padded=%d, groups=%d)\n",
           kd.col_rows, kd.col_rows_pad4, kd.num_k_groups);
    printf("                                  weights_packed: %zu int32 (%zu KB)\n",
           kd.weights_packed.size(), kd.weights_packed.size()*4/1024);
    if (layer.is_ternary)
        printf("                                  ternary dist: +1=%d (%.1f%%) 0=%d (%.1f%%) -1=%d (%.1f%%)\n",
               n_pos, 100.0*n_pos/(noc*fan_in), n_zero, 100.0*n_zero/(noc*fan_in),
               n_neg, 100.0*n_neg/(noc*fan_in));
    printf("                                  oc_scale[0..2] = [%.6f, %.6f, %.6f]\n",
           kd.oc_scale[0], kd.oc_scale.size()>1?kd.oc_scale[1]:0,
           kd.oc_scale.size()>2?kd.oc_scale[2]:0);
    printf("                                  oc_bias[0..2]  = [%.6f, %.6f, %.6f]\n",
           kd.oc_bias[0], kd.oc_bias.size()>1?kd.oc_bias[1]:0,
           kd.oc_bias.size()>2?kd.oc_bias[2]:0);
}

static void print_im2col_info(const char* tag, int col_rows, int col_cols,
                               int col_rows_pad4, int cc8, int num_k_groups,
                               const uint8_t* col_u8, const uint8_t* col_packed) {
    // col_u8 stats
    int tot_u8 = col_rows_pad4 * col_cols;
    int zeros_u8 = 0;
    for (int i = 0; i < tot_u8; i++) if (col_u8[i] == 0) zeros_u8++;
    // col_packed stats
    int tot_pk = (cc8/8) * num_k_groups * 32;

    printf("  [im2col+repack] %s\n", tag);
    printf("    col_u8:     [%d x %d] = %d bytes (%.1f KB)  zeros=%d (%.1f%%)\n",
           col_rows_pad4, col_cols, tot_u8, tot_u8/1024.0, zeros_u8, 100.0*zeros_u8/tot_u8);
    printf("    col_packed: [jg=%d x kg=%d x 32B] = %d bytes (%.1f KB)\n",
           cc8/8, num_k_groups, tot_pk, tot_pk/1024.0);
    printf("    Layout: col_packed[(jg*%d + kg)*32 + lane*4 + row_in_group]\n", num_k_groups);
    printf("            8 spatial positions x 4 reduction rows per 32-byte block\n");
    // Print first block
    printf("    First block (jg=0,kg=0): ");
    for (int i = 0; i < 32 && i < tot_pk; i++) printf("%3d ", col_packed[i]);
    printf("\n");
}

static void print_gemm_sample(const char* tag, int oc, int jg, int num_k_groups,
                                const uint8_t* col_packed, const int32_t* weights_packed,
                                float dequant, float bias) {
    // Manually compute one dot product for verification
    int32_t acc = 0;
    for (int kg = 0; kg < num_k_groups; kg++) {
        // col_packed block: 32 bytes at (jg*num_k_groups + kg)*32
        // For lane 0: bytes at offset 0,1,2,3
        const uint8_t* a = col_packed + (jg * num_k_groups + kg) * 32;
        int32_t w = weights_packed[oc * num_k_groups + kg];
        // dpbusd: for each lane, u8[lane*4+0..3] * s8[0..3] accumulated
        for (int i = 0; i < 4; i++) {
            uint8_t av = a[0*4 + i];  // lane 0
            int8_t wv = (int8_t)((w >> (i*8)) & 0xFF);
            acc += (int32_t)av * (int32_t)wv;
        }
    }
    float real = (float)acc * dequant + bias;
    printf("    %s OC=%d spatial=0: int32_acc=%d * dequant=%.6f + bias=%.4f = %.4f\n",
           tag, oc, acc, dequant, bias, real);
}

// ─── Main ───────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <model.bin> <scales.json> <test_batch>\n", argv[0]);
        return 1;
    }

    // Load model
    Model m = load_model(argv[1]);
    Engine engine;
    engine.init(m, argv[2]);

    // Load first test image
    FILE* f = fopen(argv[3], "rb");
    uint32_t count; fread(&count, 4, 1, f);
    uint32_t label; fread(&label, 4, 1, f);
    std::vector<float> img(3 * 32 * 32);
    fread(img.data(), 4, 3 * 32 * 32, f);
    fclose(f);

    Tensor input(1, 3, 32, 32);
    memcpy(input.data.data(), img.data(), 3 * 32 * 32 * sizeof(float));

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  TERNARY INFERENCE ENGINE — DEBUG WALKTHROUGH               ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    printf("Ground truth label: %d\n\n", label);

    // ── STAGE 1: Input ──
    printf("═══ STAGE 1: Input Image ═══\n");
    print_tensor_stats("input", input);
    printf("  Note: CIFAR-10 normalized with mean=[0.4914,0.4822,0.4465] std=[0.2470,0.2435,0.2616]\n");
    printf("        Values are (pixel/255 - mean) / std → range roughly [-2, +3]\n\n");

    // ── STAGE 2: First Conv (FP32) ──
    printf("═══ STAGE 2: First Conv (FP32) + BN + ReLU ═══\n");
    printf("  This is the ONLY FP32 conv. Input is float, weights are float.\n");
    printf("  BN is fused: weight*scale folded into conv weights, bias adjusted.\n");
    printf("  FP32 because: first layer sees raw floats, no quantization yet.\n\n");

    conv2d_fp32(input, engine.model.convs[0], engine.ws_fp32_a);
    auto& p0 = engine.model.convs[0].p;
    printf("  Conv params: %dch→%dch, %dx%d kernel, stride=%d, pad=%d\n",
           p0.in_channels, p0.out_channels, p0.kernel_h, p0.kernel_w, p0.stride, p0.padding);
    printf("  Optimization: AVX2 FMA (fmadd_ps) — 8 floats/cycle\n");
    printf("  im2col: FP32, [%d x %d] = %.1f KB\n",
           p0.in_channels * p0.kernel_h * p0.kernel_w, 32*32,
           p0.in_channels * p0.kernel_h * p0.kernel_w * 32.0 * 32 * 4 / 1024);
    print_tensor_stats("after conv1 (FP32 GEMM)", engine.ws_fp32_a);

    relu_inplace(engine.ws_fp32_a);
    print_tensor_stats("after ReLU", engine.ws_fp32_a);

    // ── STAGE 3: Quantize to UINT8 ──
    printf("\n═══ STAGE 3: Quantize FP32 → UINT8 ═══\n");
    printf("  All activations stay UINT8 from here until final avgpool.\n");
    printf("  Scale from calibration: real_value = uint8_value × scale\n");
    printf("  Post-ReLU → all values ≥ 0 → zero_point = 0\n");
    printf("  Optimization: AVX2 vectorized quantize (8 values/iter)\n\n");

    quantize_to_u8(engine.ws_fp32_a, engine.initial_relu_scale, engine.ws_u8_a);
    printf("  initial_relu_scale = %.6f\n", engine.initial_relu_scale);
    print_u8_stats("quantized activation", engine.ws_u8_a);

    // ── STAGE 4: ResNet Blocks ──
    printf("\n═══ STAGE 4: ResNet Blocks (INT8-Resident) ═══\n");
    printf("  8 blocks total: layer1(2) + layer2(2) + layer3(2) + layer4(2)\n");
    printf("  Each block: conv1→ReLU→conv2→residual_add→ReLU\n");
    printf("  Downsampling at layer2.0, layer3.0, layer4.0 (stride 2 + shortcut)\n\n");

    TensorU8* x = &engine.ws_u8_a;
    TensorU8* out1 = &engine.ws_u8_b;
    TensorU8* out2 = &engine.ws_u8_c;

    for (size_t bi = 0; bi < engine.blocks.size(); bi++) {
        auto& blk = engine.blocks[bi];
        auto& sc = engine.block_scales[bi];
        const char* layer_names[] = {"layer1.0","layer1.1","layer2.0","layer2.1",
                                     "layer3.0","layer3.1","layer4.0","layer4.1"};

        printf("─── Block %zu: %s ───\n", bi, layer_names[bi]);
        print_u8_stats("block input (x)", *x);

        // Show kernel info for this block's convs
        print_kernel_info("conv1 kernel", engine.model.convs[blk.conv1],
                          engine.i8_kernels[blk.conv1]);

        // Conv1
        printf("\n  ── Conv1: dpbusd GEMM (mode 1: fused ReLU + requant → u8) ──\n");
        printf("  Steps: im2col(uint8) → SSE repack(spatial-packed) → dpbusd GEMM → epilogue\n");

        auto& kd1 = engine.i8_kernels[blk.conv1];
        auto& p1 = engine.model.convs[blk.conv1].p;
        int oh1 = (x->h + 2*p1.padding - p1.kernel_h) / p1.stride + 1;
        int ow1 = (x->w + 2*p1.padding - p1.kernel_w) / p1.stride + 1;
        int cc1 = oh1 * ow1;
        int cc8_1 = (cc1 + 7) & ~7;

        printf("  Input: u8 [1,%d,%d,%d] scale=%.6f → im2col [%d x %d]\n",
               x->c, x->h, x->w, x->scale, kd1.col_rows_pad4, cc1);
        printf("  dpbusd: _mm256_dpbusd_epi32(acc, activation_u8, weight_s8_broadcast)\n");
        printf("    → 8 spatial positions × 4 reduction values per instruction = 32 MACs\n");
        printf("    → 16 OC processed per tile (16 accumulators in YMM registers)\n");
        printf("  Epilogue: int32 → float (dequant×scale+bias) → ReLU → clamp[0,255] → uint8\n");
        printf("  Output scale (calibrated): %.6f\n", sc.conv1_relu_out);

        conv2d_ternary_from_u8(*x, engine.model.convs[blk.conv1], kd1,
                               engine.ws_col_u8.data(), engine.ws_col_packed.data(),
                               1, nullptr, out1, sc.conv1_relu_out);

        // Print im2col+repack info (after the call, buffers are filled)
        print_im2col_info(layer_names[bi],
                          kd1.col_rows, cc1, kd1.col_rows_pad4, cc8_1, kd1.num_k_groups,
                          engine.ws_col_u8.data(), engine.ws_col_packed.data());
        print_gemm_sample("conv1 verify", 0, 0, kd1.num_k_groups,
                          engine.ws_col_packed.data(), kd1.weights_packed.data(),
                          x->scale * kd1.oc_scale[0], kd1.oc_bias[0]);

        print_u8_stats("conv1 output", *out1);

        // Conv2
        printf("\n  ── Conv2 + Residual ──\n");
        if (blk.shortcut >= 0) {
            printf("  Shortcut: dpbusd GEMM (mode 0 → FP32 output)\n");
            printf("    FP32 shortcut weights quantized to INT8 at model load\n");
            print_kernel_info("shortcut kernel", engine.model.convs[blk.shortcut],
                              engine.i8_kernels[blk.shortcut]);

            conv2d_ternary_from_u8(*x, engine.model.convs[blk.shortcut],
                                   engine.i8_kernels[blk.shortcut],
                                   engine.ws_col_u8.data(), engine.ws_col_packed.data(),
                                   0, &engine.ws_fp32_b, out2, 0.0f);
            print_tensor_stats("shortcut output (FP32)", engine.ws_fp32_b);

            printf("  Conv2: mode 2 — conv + add(shortcut_fp32) + ReLU + requant → u8\n");
            conv2d_ternary_from_u8(*out1, engine.model.convs[blk.conv2],
                                   engine.i8_kernels[blk.conv2],
                                   engine.ws_col_u8.data(), engine.ws_col_packed.data(),
                                   2, nullptr, out2, sc.block_relu_out,
                                   nullptr, &engine.ws_fp32_b, 0.0f);
        } else {
            printf("  Identity shortcut: residual = input x (uint8, scale=%.6f)\n", x->scale);
            printf("  Conv2: mode 2 — conv + add(x_u8) + ReLU + requant → u8\n");
            printf("    Residual add: dequant u8→float, add to conv output, ReLU, requant\n");
            conv2d_ternary_from_u8(*out1, engine.model.convs[blk.conv2],
                                   engine.i8_kernels[blk.conv2],
                                   engine.ws_col_u8.data(), engine.ws_col_packed.data(),
                                   2, nullptr, out2, sc.block_relu_out,
                                   x, nullptr, x->scale);
        }

        print_u8_stats("block output (after add+ReLU+requant)", *out2);
        printf("  Block output scale: %.6f\n\n", sc.block_relu_out);

        TensorU8* tmp = x; x = out2; out2 = tmp;
    }

    // ── STAGE 5: Avgpool + FC ──
    printf("═══ STAGE 5: Global Average Pool (uint8 → FP32) + FC ═══\n");
    printf("  Avgpool: dequant each channel (sum * scale / spatial)\n");
    printf("  FC: standard FP32 matmul (512 → 10)\n\n");

    Tensor pooled = avgpool_global_u8(*x);
    print_tensor_stats("pooled", pooled);

    auto logits = fc_forward(pooled, engine.model.fc);
    printf("  Logits: ");
    int pred = 0;
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", logits[i]);
        if (logits[i] > logits[pred]) pred = i;
    }
    printf("\n  Prediction: %d  (label: %d)  %s\n\n", pred, label,
           pred == (int)label ? "CORRECT ✓" : "WRONG ✗");

    // ── Summary ──
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  ENGINE SUMMARY                                             ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  Data types:                                                ║\n");
    printf("║    Weights: ternary {-1,0,+1} stored as int8 (2-bit packed  ║\n");
    printf("║             on disk, expanded to int8 at load time)         ║\n");
    printf("║    Activations: uint8 between all layers (post-ReLU ≥ 0)   ║\n");
    printf("║    GEMM accumulator: int32 (from dpbusd: u8×s8→i32)        ║\n");
    printf("║    Epilogue: int32→f32 (dequant) → ReLU → uint8 (requant)  ║\n");
    printf("║    Scales: float32, calibrated offline (static quantization)║\n");
    printf("║                                                             ║\n");
    printf("║  Key optimizations:                                         ║\n");
    printf("║    1. Spatial-packed dpbusd layout:                          ║\n");
    printf("║       8 spatial × 4 reduction per instruction = 32 MACs     ║\n");
    printf("║       No horizontal reduction (was 55%% port 6 saturation)   ║\n");
    printf("║    2. INT8-resident: activations stay uint8 between layers  ║\n");
    printf("║       im2col operates on bytes (4× less data than FP32)    ║\n");
    printf("║    3. Fused epilogues: residual add + ReLU + requant       ║\n");
    printf("║       inside GEMM output loop (zero intermediate tensors)  ║\n");
    printf("║    4. SSE interleaved repack: 4×8 byte transpose via       ║\n");
    printf("║       unpacklo_epi8/epi16 (4 loads + 3 shuffles + 2 stores)║\n");
    printf("║    5. 16-OC tiling: activation load reused across 16 OCs   ║\n");
    printf("║    6. Preallocated buffers: zero malloc on hot path        ║\n");
    printf("║    7. DefaultInitAlloc: skip zero-fill (GEMM writes all)   ║\n");
    printf("║    8. Persistent OMP parallel region: 1 fork/join per infer║\n");
    printf("║                                                             ║\n");
    printf("║  Memory layout (spatial-packed for dpbusd):                 ║\n");
    printf("║    col_packed[jg][kg][32 bytes]                             ║\n");
    printf("║    jg = spatial group of 8 (0..spatial/8-1)                 ║\n");
    printf("║    kg = reduction group of 4 (0..col_rows_pad4/4-1)        ║\n");
    printf("║    32 bytes = 8 lanes × 4 bytes = [s0r0,s0r1,s0r2,s0r3,   ║\n");
    printf("║                                    s1r0,s1r1,s1r2,s1r3,...]║\n");
    printf("║    dpbusd reads 32B activation + broadcasts 4B weight      ║\n");
    printf("║    Result: 8 spatial dot products accumulated per instr    ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    return 0;
}
