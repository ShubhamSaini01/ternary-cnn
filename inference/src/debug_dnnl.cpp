// Debug: compare first block conv1 output between our engine and oneDNN
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include "engine.h"
#include "engine_dnnl.h"

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "Usage: %s model scales testdata\n", argv[0]); return 1; }

    Model m = load_model(argv[1]);

    // Our engine
    Engine eng;
    eng.init(m, argv[2]);

    // oneDNN engine
    EngineDNNL eng_d;
    eng_d.init(m, argv[2]);

    // Load first image
    FILE* f = fopen(argv[3], "rb");
    uint32_t count; fread(&count, 4, 1, f);
    uint32_t label; fread(&label, 4, 1, f);
    std::vector<float> img(3*32*32);
    fread(img.data(), 4, 3*32*32, f);
    fclose(f);

    printf("Label: %u\n\n", label);

    // ── Run our engine step by step ──
    Tensor input(1, 3, 32, 32);
    memcpy(input.data.data(), img.data(), 3*32*32*sizeof(float));

    // Conv1 FP32
    conv2d_fp32(input, eng.model.convs[0], eng.ws_fp32_a);
    relu_inplace(eng.ws_fp32_a);
    printf("Our conv1+relu: [%d,%d,%d,%d] range=[%.4f, %.4f] mean=%.4f\n",
           eng.ws_fp32_a.n, eng.ws_fp32_a.c, eng.ws_fp32_a.h, eng.ws_fp32_a.w,
           *std::min_element(eng.ws_fp32_a.data.begin(), eng.ws_fp32_a.data.begin()+eng.ws_fp32_a.total()),
           *std::max_element(eng.ws_fp32_a.data.begin(), eng.ws_fp32_a.data.begin()+eng.ws_fp32_a.total()),
           [&]{ double s=0; for(int i=0;i<eng.ws_fp32_a.total();i++) s+=eng.ws_fp32_a.data[i]; return s/eng.ws_fp32_a.total(); }());

    // Quantize
    quantize_to_u8(eng.ws_fp32_a, eng.initial_relu_scale, eng.ws_u8_a);
    printf("Our quantized: scale=%.6f range=[%d, %d]\n",
           eng.ws_u8_a.scale,
           *std::min_element(eng.ws_u8_a.data.begin(), eng.ws_u8_a.data.begin()+eng.ws_u8_a.total()),
           *std::max_element(eng.ws_u8_a.data.begin(), eng.ws_u8_a.data.begin()+eng.ws_u8_a.total()));

    // Block 0 conv1
    auto& blk = eng.blocks[0];
    auto& sc = eng.block_scales[0];
    auto& kd1 = eng.i8_kernels[blk.conv1];
    conv2d_ternary_from_u8(eng.ws_u8_a, eng.model.convs[blk.conv1], kd1,
                           eng.ws_col_u8.data(), eng.ws_col_packed.data(),
                           1, nullptr, &eng.ws_u8_b, sc.conv1_relu_out);
    printf("Our block0.conv1: scale=%.6f range=[%d, %d] first8=[",
           eng.ws_u8_b.scale,
           *std::min_element(eng.ws_u8_b.data.begin(), eng.ws_u8_b.data.begin()+eng.ws_u8_b.total()),
           *std::max_element(eng.ws_u8_b.data.begin(), eng.ws_u8_b.data.begin()+eng.ws_u8_b.total()));
    for(int i=0;i<8;i++) printf("%d ", eng.ws_u8_b.data[i]);
    printf("]\n");

    // Full forward for final result
    auto our_logits = eng.forward(input);
    printf("Our logits: ");
    int our_pred = 0;
    for(int i=0;i<10;i++) { printf("%.2f ", our_logits[i]); if(our_logits[i]>our_logits[our_pred]) our_pred=i; }
    printf("→ %d\n\n", our_pred);

    // ── Run oneDNN engine ──
    auto dnnl_logits = eng_d.forward(img);
    printf("DNNL logits: ");
    int dnnl_pred = 0;
    for(int i=0;i<10;i++) { printf("%.2f ", dnnl_logits[i]); if(dnnl_logits[i]>dnnl_logits[dnnl_pred]) dnnl_pred=i; }
    printf("→ %d\n\n", dnnl_pred);

    // Compare
    double max_diff = 0;
    for(int i=0;i<10;i++) max_diff = std::max(max_diff, std::abs((double)our_logits[i] - dnnl_logits[i]));
    printf("Max logit diff: %.4f\n", max_diff);
    printf("Our pred=%d  DNNL pred=%d  Label=%u\n", our_pred, dnnl_pred, label);

    return 0;
}
