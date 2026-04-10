#pragma once
// ═══════════════════════════════════════════════════════════════
// Hybrid Engine: oneDNN JIT convolution + custom memory/threading
//
// Uses oneDNN convolution primitives for ternary INT8 layers
// (replacing hand-written im2col + VNNI interleave + dpbusd),
// while keeping custom:
//   - Dynamic FP32→u8 quantization
//   - s32→FP32 dequant with fused BN scale/bias + residual + ReLU
//   - FP32 conv (stem, shortcuts)
//   - Memory pre-allocation (oneDNN memory objects + buffers)
//   - Thread count control via OMP
//
// This is the "ONNX Runtime approach": oneDNN primitives with
// preallocated memory objects and minimal synchronization.
// ═══════════════════════════════════════════════════════════════

#include "ternary_engine.h"   // reuse Tensor, TernaryKernel, FP32 ops, etc.
#include <dnnl.hpp>

// ─── oneDNN primitive cache per ternary layer ────────────────
struct DnnlConvPrimitive {
    dnnl::convolution_forward prim;
    dnnl::memory weights_mem;          // pre-reordered, persistent
    dnnl::memory::desc src_md;         // NCHW u8
    dnnl::memory::desc dst_md;         // NCHW s32
    int in_c, in_h, in_w;
    int out_c, out_h, out_w;
    bool valid = false;
};

// ─── Hybrid profile ─────────────────────────────────────────
static EngineProfile g_hybrid_prof;

// ─── Quantize FP32 → u8 (NCHW, no manual padding) ──────────
static float quantize_fp32_to_u8(const Tensor& input, int ni,
                                  uint8_t* u8_buf) {
    int C = input.c, H = input.h, W = input.w;
    int tot = C * H * W;
    const float* src = input.data.data() + ni * tot;

    float gmax = 0.0f;
    int i = 0;
#ifdef __AVX2__
    __m256 vm = _mm256_setzero_ps(), sm = _mm256_set1_ps(-0.0f);
    for (; i + 7 < tot; i += 8) {
        __m256 v = _mm256_andnot_ps(sm, _mm256_loadu_ps(src + i));
        vm = _mm256_max_ps(vm, v);
    }
    __m128 hh = _mm256_extractf128_ps(vm, 1);
    __m128 ll = _mm256_castps256_ps128(vm);
    __m128 m = _mm_max_ps(ll, hh);
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(2, 3, 0, 1)));
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(1, 0, 3, 2)));
    gmax = _mm_cvtss_f32(m);
#endif
    for (; i < tot; i++) {
        float v = fabsf(src[i]);
        if (v > gmax) gmax = v;
    }
    if (gmax < 1e-10f) {
        memset(u8_buf, 128, tot);
        return 0.0f;
    }

    float qscale = 127.0f / gmax;
    i = 0;
#ifdef __AVX2__
    __m256 qv = _mm256_set1_ps(qscale);
    __m256 off = _mm256_set1_ps(128.0f);
    __m256 lo = _mm256_setzero_ps();
    __m256 hi = _mm256_set1_ps(255.0f);
    for (; i + 7 < tot; i += 8) {
        __m256 v = _mm256_fmadd_ps(_mm256_loadu_ps(src + i), qv, off);
        v = _mm256_min_ps(_mm256_max_ps(v, lo), hi);
        __m256i i32 = _mm256_cvtps_epi32(v);
        __m256i p16 = _mm256_packs_epi32(i32, i32);
        p16 = _mm256_permute4x64_epi64(p16, 0xD8);
        __m128i p8 = _mm_packus_epi16(
            _mm256_castsi256_si128(p16), _mm256_castsi256_si128(p16));
        _mm_storel_epi64((__m128i*)(u8_buf + i), p8);
    }
#endif
    for (; i < tot; i++) {
        int v = (int)roundf(src[i] * qscale + 128.0f);
        u8_buf[i] = (uint8_t)(v > 255 ? 255 : (v < 0 ? 0 : v));
    }
    return gmax / 127.0f;
}

// ─── Convert INT8Buffer (s8) → u8 ──────────────────────────
static float convert_s8_to_u8(const INT8Buffer& buf, uint8_t* u8_buf) {
    int tot = buf.c * buf.h * buf.w;
    int i = 0;
#ifdef __AVX2__
    __m256i bias = _mm256_set1_epi8((char)0x80);
    for (; i + 31 < tot; i += 32) {
        __m256i v = _mm256_loadu_si256((__m256i*)(buf.data.data() + i));
        _mm256_storeu_si256((__m256i*)(u8_buf + i),
                            _mm256_xor_si256(v, bias));
    }
#endif
    for (; i < tot; i++)
        u8_buf[i] = (uint8_t)((int)buf.data[i] + 128);
    return buf.inv_scale;
}

// ─── Hybrid ternary conv using oneDNN ───────────────────────
static void dnnl_conv2d_ternary(
    const Tensor* input_fp32, const INT8Buffer* input_i8,
    const TernaryConvLayer& layer, TernaryKernel& kernel,
    DnnlConvPrimitive& dc,
    dnnl::engine& eng, dnnl::stream& strm,
    std::vector<uint8_t>& u8_src_buf,
    std::vector<int32_t>& s32_dst_buf,
    Tensor& output_fp32, INT8Buffer* output_i8,
    const Tensor* residual, bool apply_relu)
{
    PTimer pt;
    auto& p = layer.params;
    int batch = input_fp32 ? input_fp32->n : 1;
    int out_h = dc.out_h, out_w = dc.out_w;
    int col_cols = out_h * out_w;

    output_fp32.reshape(batch, p.out_channels, out_h, out_w);
    if (output_i8) output_i8->reshape(p.out_channels, out_h, out_w);

    for (int ni = 0; ni < batch; ni++) {
        float inv_scale;

        // ── Step 1: Quantize to u8 ──
        pt.start();
        if (input_fp32)
            inv_scale = quantize_fp32_to_u8(*input_fp32, ni, u8_src_buf.data());
        else
            inv_scale = convert_s8_to_u8(*input_i8, u8_src_buf.data());
#if PROFILE_ENGINE
        g_hybrid_prof.t_quantize += pt.stop_ms();
#endif

        // ── Step 2: oneDNN convolution (u8 × s8 → s32) ──
        pt.start();
        {
            dnnl::memory src_mem(dc.src_md, eng, u8_src_buf.data());
            dnnl::memory dst_mem(dc.dst_md, eng, s32_dst_buf.data());
            dc.prim.execute(strm, {
                {DNNL_ARG_SRC, src_mem},
                {DNNL_ARG_WEIGHTS, dc.weights_mem},
                {DNNL_ARG_DST, dst_mem}
            });
            strm.wait();
        }
#if PROFILE_ENGINE
        g_hybrid_prof.t_im2col += pt.stop_ms();  // reuse counter as "dnnl_compute"
#endif

        // ── Step 3: Dequant s32 → FP32 + BN + residual + ReLU ──
        pt.start();
        for (uint32_t oc = 0; oc < p.out_channels; oc++) {
            float combined = inv_scale * layer.fused_scale[oc];
            float adj_bias = layer.fused_bias[oc]
                             - 128.0f * (float)kernel.weight_sum[oc] * combined;
            const int32_t* acc = s32_dst_buf.data() + oc * col_cols;
            float* out_row = output_fp32.channel_ptr(ni, oc);
            const float* res_row = residual
                ? residual->channel_ptr(ni, oc) : nullptr;
            dequant_oc(acc, out_row, res_row, col_cols,
                       combined, adj_bias, apply_relu);
        }
#if PROFILE_ENGINE
        g_hybrid_prof.t_compute += pt.stop_ms();
        g_hybrid_prof.ternary_calls++;
#endif

        // ── INT8 output requantization ──
        if (output_i8) {
            int total_el = p.out_channels * col_cols;
            const float* src = output_fp32.data.data();
            float gmax = 0.0f;
            int i = 0;
#ifdef __AVX2__
            __m256 vm2 = _mm256_setzero_ps(), sm2 = _mm256_set1_ps(-0.0f);
            for (; i + 7 < total_el; i += 8) {
                __m256 v = _mm256_andnot_ps(sm2, _mm256_loadu_ps(src + i));
                vm2 = _mm256_max_ps(vm2, v);
            }
            __m128 hh2 = _mm256_extractf128_ps(vm2, 1);
            __m128 ll2 = _mm256_castps256_ps128(vm2);
            __m128 mm2 = _mm_max_ps(ll2, hh2);
            mm2 = _mm_max_ps(mm2, _mm_shuffle_ps(mm2, mm2, _MM_SHUFFLE(2, 3, 0, 1)));
            mm2 = _mm_max_ps(mm2, _mm_shuffle_ps(mm2, mm2, _MM_SHUFFLE(1, 0, 3, 2)));
            gmax = _mm_cvtss_f32(mm2);
#endif
            for (; i < total_el; i++) {
                float v = fabsf(src[i]);
                if (v > gmax) gmax = v;
            }
            if (gmax < 1e-10f) {
                memset(output_i8->data.data(), 0, (size_t)total_el);
                output_i8->inv_scale = 0.0f;
            } else {
                float qscale = 127.0f / gmax;
                output_i8->inv_scale = gmax / 127.0f;
                int8_t* dst = output_i8->data.data();
                i = 0;
#ifdef __AVX2__
                __m256 qv = _mm256_set1_ps(qscale);
                for (; i + 7 < total_el; i += 8) {
                    __m256 v = _mm256_mul_ps(_mm256_loadu_ps(src + i), qv);
                    __m256i i32 = _mm256_cvtps_epi32(v);
                    __m256i p16 = _mm256_packs_epi32(i32, i32);
                    p16 = _mm256_permute4x64_epi64(p16, 0xD8);
                    _mm_storel_epi64((__m128i*)(dst + i),
                        _mm_packs_epi16(_mm256_castsi256_si128(p16),
                                        _mm256_castsi256_si128(p16)));
                }
#endif
                for (; i < total_el; i++) {
                    int v = (int)roundf(src[i] * qscale);
                    dst[i] = (int8_t)(v > 127 ? 127 : (v < -128 ? -128 : v));
                }
            }
        }
    }
}


// ═══════════════════════════════════════════════════════════════
// HybridTernaryResNet18
// ═══════════════════════════════════════════════════════════════
class HybridTernaryResNet18 {
public:
    Model model;
    std::vector<TernaryKernel> ternary_kernels;
    std::vector<DnnlConvPrimitive> dnnl_convs;

    dnnl::engine eng;
    dnnl::stream strm;

    // Pre-allocated buffers
    std::vector<float>   col_fp32_buf;
    std::vector<uint8_t> u8_src_buf;
    std::vector<int32_t> s32_dst_buf;
    Tensor work_a, work_b, work_sc;
    INT8Buffer int8_mid;

    bool load(const std::string& path) {
        if (!model.load(path)) return false;

        eng = dnnl::engine(dnnl::engine::kind::cpu, 0);
        strm = dnnl::stream(eng);

        printf("Preparing Hybrid oneDNN+Custom engine...\n");

        ternary_kernels.resize(model.layers.size());
        dnnl_convs.resize(model.layers.size());

        int max_fp32_col = 0, max_u8_src = 0, max_s32_dst = 0, max_out = 0;

        // ── Compute spatial dims by walking the ResNet-18 topology ──
        // Layer order: stem, then 4 groups × 2 blocks, then FC
        // basic_block reads: ternary1, ternary2, [shortcut if has_sc]

        int h = 32, w = 32;
        int li = 0; // layer index

        // Stem (FP32 conv)
        {
            auto& p = model.layers[li].fp32_conv.params;
            int cr = p.in_channels * p.kernel_h * p.kernel_w;
            int oh = (h + 2*p.padding - p.kernel_h) / p.stride + 1;
            int ow = (w + 2*p.padding - p.kernel_w) / p.stride + 1;
            if (cr * oh * ow > max_fp32_col) max_fp32_col = cr * oh * ow;
            h = oh; w = ow;
            li++;
        }

        // 8 basic blocks: groups [0..3], 2 blocks each
        // has_sc at first block of groups 1,2,3
        for (int grp = 0; grp < 4; grp++) {
            for (int blk = 0; blk < 2; blk++) {
                bool has_sc = (grp > 0 && blk == 0);
                int block_h = h, block_w = w;

                // First ternary conv
                {
                    int idx = li++;
                    auto& lay = model.layers[idx].ternary_conv;
                    auto& p = lay.params;
                    ternary_kernels[idx].prepare(lay);
                    create_dnnl_conv(idx, lay, ternary_kernels[idx], h, w);
                    int u8sz = p.in_channels * h * w;
                    int oh = (h + 2*p.padding - p.kernel_h) / p.stride + 1;
                    int ow = (w + 2*p.padding - p.kernel_w) / p.stride + 1;
                    int s32sz = p.out_channels * oh * ow;
                    if (u8sz > max_u8_src) max_u8_src = u8sz;
                    if (s32sz > max_s32_dst) max_s32_dst = s32sz;
                    if (s32sz > max_out) max_out = s32sz;
                    h = oh; w = ow;
                }

                // Second ternary conv
                {
                    int idx = li++;
                    auto& lay = model.layers[idx].ternary_conv;
                    auto& p = lay.params;
                    ternary_kernels[idx].prepare(lay);
                    create_dnnl_conv(idx, lay, ternary_kernels[idx], h, w);
                    int u8sz = p.in_channels * h * w;
                    int oh = (h + 2*p.padding - p.kernel_h) / p.stride + 1;
                    int ow = (w + 2*p.padding - p.kernel_w) / p.stride + 1;
                    int s32sz = p.out_channels * oh * ow;
                    if (u8sz > max_u8_src) max_u8_src = u8sz;
                    if (s32sz > max_s32_dst) max_s32_dst = s32sz;
                    if (s32sz > max_out) max_out = s32sz;
                    h = oh; w = ow;
                }

                // Shortcut (FP32 1×1)
                if (has_sc) {
                    auto& p = model.layers[li].fp32_conv.params;
                    int cr = p.in_channels * p.kernel_h * p.kernel_w;
                    int oh = (block_h + 2*p.padding - p.kernel_h) / p.stride + 1;
                    int ow = (block_w + 2*p.padding - p.kernel_w) / p.stride + 1;
                    if (cr * oh * ow > max_fp32_col) max_fp32_col = cr * oh * ow;
                    li++;
                }
            }
        }

        // Allocate buffers
        col_fp32_buf.resize(max_fp32_col);
        u8_src_buf.resize(max_u8_src);
        s32_dst_buf.resize(max_s32_dst);
        work_a.data.resize(max_out);
        work_b.data.resize(max_out);
        work_sc.data.resize(max_out);
        int8_mid.data.resize(max_out);

        printf("Pre-allocated: u8_src=%.1fKB s32_dst=%.1fKB\n",
               max_u8_src / 1024.0, max_s32_dst * 4 / 1024.0);
        printf("Hybrid oneDNN+Custom engine ready.\n\n");
        return true;
    }

    Tensor forward(const Tensor& input) {
        int idx = 0;
        conv2d_fp32_into(input, model.layers[idx++].fp32_conv,
                         col_fp32_buf, work_a);
        relu_inplace(work_a);

        basic_block(work_a, work_b, idx, false);
        basic_block(work_b, work_a, idx, false);
        basic_block(work_a, work_b, idx, true);
        basic_block(work_b, work_a, idx, false);
        basic_block(work_a, work_b, idx, true);
        basic_block(work_b, work_a, idx, false);
        basic_block(work_a, work_b, idx, true);
        basic_block(work_b, work_a, idx, false);

        Tensor pooled = global_avg_pool(work_a);
        return fc_forward(pooled, model.layers[idx++].fc);
    }

    void print_profile() {
        auto& p = g_hybrid_prof;
        double t_tern = p.t_quantize + p.t_im2col + p.t_compute;
        double total = t_tern + p.t_fp32_total + p.t_relu + p.t_add + p.t_pool + p.t_fc;
        printf("\n═══════════════════════════════════════════════════\n");
        printf("HYBRID ENGINE PROFILE (oneDNN conv + custom dequant)\n");
        printf("═══════════════════════════════════════════════════\n\n");
        printf("Ternary Conv (%d calls):           %8.3f ms\n", p.ternary_calls, t_tern);
        printf("  ├─ quantize FP32→u8:             %8.3f ms  (%4.1f%%)\n", p.t_quantize, 100*p.t_quantize/total);
        printf("  ├─ oneDNN conv (u8×s8→s32):      %8.3f ms  (%4.1f%%)\n", p.t_im2col, 100*p.t_im2col/total);
        printf("  └─ dequant+BN+ReLU:              %8.3f ms  (%4.1f%%)\n", p.t_compute, 100*p.t_compute/total);
        printf("\nFP32 Conv (%d calls):              %8.3f ms  (%4.1f%%)\n", p.fp32_calls, p.t_fp32_total, 100*p.t_fp32_total/total);
        printf("\nOther:\n");
        printf("  ├─ ReLU:                         %8.3f ms  (%4.1f%%)\n", p.t_relu, 100*p.t_relu/total);
        printf("  ├─ Add:                          %8.3f ms  (%4.1f%%)\n", p.t_add, 100*p.t_add/total);
        printf("  ├─ Pool:                         %8.3f ms  (%4.1f%%)\n", p.t_pool, 100*p.t_pool/total);
        printf("  └─ FC:                           %8.3f ms  (%4.1f%%)\n", p.t_fc, 100*p.t_fc/total);
        printf("\n───────────────────────────────────────────────────\n");
        printf("TOTAL:                             %8.3f ms\n", total);
        printf("═══════════════════════════════════════════════════\n");
    }
    void reset_profile() { g_hybrid_prof.reset(); }

private:
    void create_dnnl_conv(int layer_idx,
                          const TernaryConvLayer& layer,
                          TernaryKernel& kernel,
                          int in_h, int in_w) {
        using namespace dnnl;
        auto& p = layer.params;
        int out_h = (in_h + 2*p.padding - p.kernel_h) / p.stride + 1;
        int out_w = (in_w + 2*p.padding - p.kernel_w) / p.stride + 1;

        memory::dims src_dims = {1, (memory::dim)p.in_channels, in_h, in_w};
        memory::dims wt_dims  = {(memory::dim)p.out_channels,
                                 (memory::dim)p.in_channels,
                                 (memory::dim)p.kernel_h,
                                 (memory::dim)p.kernel_w};
        memory::dims dst_dims = {1, (memory::dim)p.out_channels, out_h, out_w};
        memory::dims strides_d = {(memory::dim)p.stride, (memory::dim)p.stride};
        memory::dims pad_l = {(memory::dim)p.padding, (memory::dim)p.padding};
        memory::dims pad_r = {(memory::dim)p.padding, (memory::dim)p.padding};

        // User formats: NCHW for src/dst (matches our Tensor layout)
        auto src_md = memory::desc(src_dims, memory::data_type::u8,
                                   memory::format_tag::nchw);
        auto dst_md = memory::desc(dst_dims, memory::data_type::s32,
                                   memory::format_tag::nchw);
        // Let oneDNN choose optimal weight layout
        auto wt_any = memory::desc(wt_dims, memory::data_type::s8,
                                   memory::format_tag::any);

        // Primitive descriptor — no bias
        auto conv_pd = convolution_forward::primitive_desc(
            eng, prop_kind::forward_inference,
            algorithm::convolution_direct,
            src_md, wt_any, dst_md,
            strides_d, pad_l, pad_r);

        // Prepare weights: copy from unpacked [OC][col_rows_pad4] → [OC][IC][KH][KW]
        auto user_wt_md = memory::desc(wt_dims, memory::data_type::s8,
                                       memory::format_tag::oihw);
        auto user_wt_mem = memory(user_wt_md, eng);
        int8_t* w_ptr = (int8_t*)user_wt_mem.get_data_handle();
        int col_rows = p.in_channels * p.kernel_h * p.kernel_w;
        for (uint32_t oc = 0; oc < p.out_channels; oc++) {
            const int8_t* src_w = kernel.unpacked_weights.data()
                                  + oc * kernel.col_rows_pad4;
            memcpy(w_ptr + oc * col_rows, src_w, col_rows);
        }

        // Reorder weights to oneDNN's preferred format
        auto& dc = dnnl_convs[layer_idx];
        dc.weights_mem = memory(conv_pd.weights_desc(), eng);
        reorder(user_wt_mem, dc.weights_mem).execute(strm,
            user_wt_mem, dc.weights_mem);
        strm.wait();

        dc.prim = convolution_forward(conv_pd);
        dc.src_md = src_md;
        dc.dst_md = dst_md;
        dc.in_c = p.in_channels;  dc.in_h = in_h;  dc.in_w = in_w;
        dc.out_c = p.out_channels; dc.out_h = out_h; dc.out_w = out_w;
        dc.valid = true;

        printf("  oneDNN conv: %ux%u [%dx%d] s=%u → %ux%u [%dx%d]\n",
               p.in_channels, p.out_channels, in_h, in_w,
               p.stride, p.out_channels, p.out_channels, out_h, out_w);
    }

    void basic_block(Tensor& x_in, Tensor& x_out, int& idx, bool has_sc) {
        int idx1 = idx++, idx2 = idx++, idx_sc = has_sc ? idx++ : -1;

        // First ternary conv → FP32 + INT8 intermediate
        dnnl_conv2d_ternary(
            &x_in, nullptr,
            model.layers[idx1].ternary_conv, ternary_kernels[idx1],
            dnnl_convs[idx1], eng, strm,
            u8_src_buf, s32_dst_buf,
            work_sc, &int8_mid, nullptr, true);

        // Shortcut (FP32 1×1 conv)
        if (has_sc)
            conv2d_fp32_into(x_in, model.layers[idx_sc].fp32_conv,
                             col_fp32_buf, work_sc);

        const Tensor* res = has_sc ? &work_sc : &x_in;

        // Second ternary conv from INT8 intermediate
        dnnl_conv2d_ternary(
            nullptr, &int8_mid,
            model.layers[idx2].ternary_conv, ternary_kernels[idx2],
            dnnl_convs[idx2], eng, strm,
            u8_src_buf, s32_dst_buf,
            x_out, nullptr, res, true);
    }
};
