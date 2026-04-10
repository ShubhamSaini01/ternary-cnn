#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include "oneapi/dnnl/dnnl.hpp"
#include "model_loader.h"

// ═══════════════════════════════════════════════════════════════
// oneDNN-backed INT8 ResNet-18 Inference Engine
//
// Replaces hand-rolled im2col+repack+dpbusd with oneDNN's fused
// convolution primitives. Same ternary weights (as INT8), same
// BN fusion, same static scales.
//
// oneDNN handles: optimal memory layout, fused micro-kernels,
// register tiling, cache blocking — the 30% overhead we couldn't
// eliminate by hand.
// ═══════════════════════════════════════════════════════════════

static void write_to_dnnl_memory(const void* handle, dnnl::memory& mem) {
    uint8_t* dst = static_cast<uint8_t*>(mem.get_data_handle());
    if (!dst) return;
    size_t size = mem.get_desc().get_size();
    memcpy(dst, handle, size);
}

static void read_from_dnnl_memory(void* handle, const dnnl::memory& mem) {
    const uint8_t* src = static_cast<const uint8_t*>(mem.get_data_handle());
    size_t size = mem.get_desc().get_size();
    memcpy(handle, src, size);
}

// ─── Per-layer conv config ──────────────────────────────────

struct DnnlConvLayer {
    std::string name;
    bool is_ternary;

    // Primitive + memory (created at init, reused at inference)
    dnnl::convolution_forward conv_prim;
    dnnl::convolution_forward::primitive_desc conv_pd;

    dnnl::memory weights_mem;   // s8, in oneDNN-chosen layout
    dnnl::memory bias_mem;      // f32 fused bias

    // Scale memories
    dnnl::memory src_scale_mem;
    dnnl::memory wei_scale_mem;
    dnnl::memory dst_scale_mem;

    // Dimensions for output
    int oc, oh, ow;
};

// ─── Engine ─────────────────────────────────────────────────

class EngineDNNL {
public:
    dnnl::engine eng;
    dnnl::stream strm;
    Model model;

    // Conv layers in execution order
    // conv[0] = first conv (FP32→u8)
    // conv[1..] = ternary/shortcut convs
    std::vector<DnnlConvLayer> layers;

    // Block structure
    struct BlockInfo {
        int conv1_idx, conv2_idx, shortcut_idx;  // indices into layers[]
    };
    std::vector<BlockInfo> blocks;

    // Activation memories — ping-pong between blocks
    dnnl::memory act_mem;  // current activation (u8)

    // Scales from calibration
    float initial_relu_scale;
    struct ScaleSet {
        float conv1_relu_out;
        float conv2_bn_out;
        float shortcut_out;
        float block_relu_out;
    };
    std::vector<ScaleSet> block_scales;

    // FC layer
    dnnl::inner_product_forward fc_prim;
    dnnl::inner_product_forward::primitive_desc fc_pd;
    dnnl::memory fc_weights_mem, fc_bias_mem;

    void init(const Model& m, const char* scales_path) {
        model = m;
        eng = dnnl::engine(dnnl::engine::kind::cpu, 0);
        strm = dnnl::stream(eng);

        load_scales(scales_path);

        // Build block structure
        int idx = 1;
        bool has_sc[] = {false, true, true, true};
        for (int l = 0; l < 4; l++)
            for (int b = 0; b < 2; b++) {
                BlockInfo bi;
                bi.conv1_idx = idx++;
                bi.conv2_idx = idx++;
                bi.shortcut_idx = (b == 0 && has_sc[l]) ? idx++ : -1;
                blocks.push_back(bi);
            }

        // Create all conv primitives
        layers.resize(model.convs.size());
        create_first_conv();
        for (size_t bi = 0; bi < blocks.size(); bi++)
            create_block_convs(bi);
        create_fc();

        printf("EngineDNNL: %zu conv layers, %zu blocks, oneDNN backend\n",
               model.convs.size(), blocks.size());
    }

    // ── Scale computation ──
    // oneDNN scaling: output_u8 = (src_scale * wei_scale / dst_scale) * int32_acc + bias_adjusted
    // Our model: real_output = acc * act_scale * alpha * bn_scale + bn_bias
    //            u8_output = real_output / out_scale
    //
    // For ternary: weights are {-1,0,+1} as s8, alpha is per-channel.
    //   int32_acc = sum(u8_act * s8_weight)
    //   real = int32_acc * act_scale * alpha[oc] * bn_scale[oc] + bn_bias[oc]
    //   u8_out = real / out_act_scale
    //
    // oneDNN wants: u8_out = (src_sc * wei_sc / dst_sc) * int32_acc + bias_in_output_scale
    //   src_sc = act_scale (scalar)
    //   wei_sc[oc] = alpha[oc] * bn_scale[oc] (per-channel)
    //   dst_sc = 1/out_act_scale (inverse, because oneDNN divides by dst_scale)
    //   bias[oc] = bn_bias[oc] (in real units, oneDNN applies scales automatically)

    // First conv produces f32 → we quantize to u8 manually in forward()
    dnnl::convolution_forward conv0_prim;
    dnnl::convolution_forward::primitive_desc conv0_pd;
    dnnl::memory conv0_weights_mem, conv0_bias_mem;
    int conv0_oc, conv0_oh, conv0_ow;

    void create_first_conv() {
        auto& fc = model.convs[0];
        auto& p = fc.p;
        conv0_oh = (32 + 2 * p.padding - p.kernel_h) / p.stride + 1;
        conv0_ow = (32 + 2 * p.padding - p.kernel_w) / p.stride + 1;
        conv0_oc = p.out_channels;

        using namespace dnnl;
        using dt = memory::data_type;
        using ft = memory::format_tag;

        auto src_md = memory::desc({1, 3, 32, 32}, dt::f32, ft::nchw);
        auto wei_md = memory::desc({p.out_channels, 3, p.kernel_h, p.kernel_w}, dt::f32, ft::any);
        auto bias_md = memory::desc({p.out_channels}, dt::f32, ft::x);
        auto dst_md = memory::desc({1, p.out_channels, conv0_oh, conv0_ow}, dt::f32, ft::any);

        // Fused ReLU post-op
        post_ops ops;
        ops.append_eltwise(algorithm::eltwise_relu, 0.f, 0.f);
        primitive_attr attr;
        attr.set_post_ops(ops);

        conv0_pd = convolution_forward::primitive_desc(eng, prop_kind::forward_inference,
            algorithm::convolution_direct, src_md, wei_md, bias_md, dst_md,
            {p.stride, p.stride}, {p.padding, p.padding}, {p.padding, p.padding}, attr);

        conv0_weights_mem = memory(conv0_pd.weights_desc(), eng);
        {
            auto user_wei = memory({{p.out_channels, 3, p.kernel_h, p.kernel_w}, dt::f32, ft::oihw}, eng);
            write_to_dnnl_memory(fc.weights_fp32.data(), user_wei);
            reorder(user_wei, conv0_weights_mem).execute(strm, user_wei, conv0_weights_mem);
            strm.wait();
        }

        conv0_bias_mem = memory(conv0_pd.bias_desc(), eng);
        write_to_dnnl_memory(fc.fused_bias.data(), conv0_bias_mem);

        conv0_prim = convolution_forward(conv0_pd);
    }

    // Create a single int8 conv primitive (ternary or shortcut)
    // epilogue: 0=f32 output (no relu), 1=relu→u8, 2=sum+relu→u8
    // sum_input_scale: for epilogue 2, the scale of the residual in the dst buffer
    void create_int8_conv(int conv_idx, int ic, int ih, int iw,
                          float src_scale, float dst_scale,
                          int epilogue, float sum_input_scale = 0.0f) {
        using namespace dnnl;
        using dt = memory::data_type;
        using ft = memory::format_tag;

        auto& fc = model.convs[conv_idx];
        auto& p = fc.p;
        int oh = (ih + 2 * p.padding - p.kernel_h) / p.stride + 1;
        int ow = (iw + 2 * p.padding - p.kernel_w) / p.stride + 1;

        auto src_md = memory::desc({1, ic, ih, iw}, dt::u8, ft::any);
        auto wei_md = memory::desc({p.out_channels, ic, p.kernel_h, p.kernel_w}, dt::s8, ft::any);
        auto bias_md = memory::desc({p.out_channels}, dt::f32, ft::x);
        // f32 output for mode 0, u8 for modes 1 and 2
        auto dst_md = (epilogue == 0)
            ? memory::desc({1, p.out_channels, oh, ow}, dt::f32, ft::any)
            : memory::desc({1, p.out_channels, oh, ow}, dt::u8, ft::any);

        primitive_attr attr;

        // Source scale (scalar)
        attr.set_scales_mask(DNNL_ARG_SRC, 0);

        // Weight scale (per-channel)
        attr.set_scales_mask(DNNL_ARG_WEIGHTS, 1 << 0);

        // Destination scale (only for u8 output)
        if (epilogue != 0)
            attr.set_scales_mask(DNNL_ARG_DST, 0);

        // Post-ops
        post_ops ops;
        if (epilogue == 2) {
            // Sum post-op: f32_result = conv_f32 + sum_scale * (float)old_dst_u8
            // Then: u8_out = relu(f32_result) / dst_scale
            // We want residual contribution = in_scale * old_u8
            // So sum_scale = in_scale (the residual's quantization scale)
            ops.append_sum(sum_input_scale);
        }
        if (epilogue != 0)
            ops.append_eltwise(algorithm::eltwise_relu, 0.f, 0.f);
        attr.set_post_ops(ops);

        auto pd = convolution_forward::primitive_desc(eng, prop_kind::forward_inference,
            algorithm::convolution_direct, src_md, wei_md, bias_md, dst_md,
            {p.stride, p.stride}, {p.padding, p.padding}, {p.padding, p.padding}, attr);

        // Prepare weights as s8
        auto wei_mem = memory(pd.weights_desc(), eng);
        {
            // Weights: either ternary i8 or FP32→i8 for shortcuts
            std::vector<int8_t> w_i8;
            std::vector<float> w_scales(p.out_channels);

            if (fc.is_ternary) {
                w_i8 = fc.weights_i8;
                for (int o = 0; o < p.out_channels; o++)
                    w_scales[o] = fc.alpha[o] * fc.fused_scale[o];
            } else {
                // Shortcut: quantize FP32 → s8 per-channel
                int fan_in = ic * p.kernel_h * p.kernel_w;
                w_i8.resize(p.out_channels * fan_in);
                for (int o = 0; o < p.out_channels; o++) {
                    const float* w = fc.weights_fp32.data() + o * fan_in;
                    float maxabs = 0;
                    for (int k = 0; k < fan_in; k++) maxabs = std::max(maxabs, fabsf(w[k]));
                    float ws = (maxabs > 1e-10f) ? maxabs / 127.0f : 1.0f;
                    float inv = 1.0f / ws;
                    for (int k = 0; k < fan_in; k++)
                        w_i8[o * fan_in + k] = (int8_t)std::max(-128, std::min(127, (int)roundf(w[k] * inv)));
                    w_scales[o] = ws * fc.fused_scale[o];
                }
            }

            auto user_wei = memory({{p.out_channels, ic, p.kernel_h, p.kernel_w}, dt::s8, ft::oihw}, eng);
            write_to_dnnl_memory(w_i8.data(), user_wei);
            reorder(user_wei, wei_mem).execute(strm, user_wei, wei_mem);
            strm.wait();

            // Store per-channel weight scales
            auto ws_md = memory::desc({p.out_channels}, dt::f32, ft::x);
            layers[conv_idx].wei_scale_mem = memory(ws_md, eng);
            write_to_dnnl_memory(w_scales.data(), layers[conv_idx].wei_scale_mem);
        }

        // Bias (fused BN bias)
        auto bias_mem = memory(pd.bias_desc(), eng);
        write_to_dnnl_memory(fc.fused_bias.data(), bias_mem);

        // Source scale
        float src_sc_val = src_scale;
        auto src_sc = memory({{1}, dt::f32, ft::x}, eng);
        write_to_dnnl_memory(&src_sc_val, src_sc);

        // Dst scale: oneDNN computes u8_out = f32_result / dst_scale
        // dst_scale = output activation scale (maps u8 → real: real = u8 * scale)
        auto dst_sc = memory({{1}, dt::f32, ft::x}, eng);
        write_to_dnnl_memory(&dst_scale, dst_sc);

        layers[conv_idx].name = fc.name;
        layers[conv_idx].is_ternary = fc.is_ternary;
        layers[conv_idx].conv_pd = pd;
        layers[conv_idx].conv_prim = convolution_forward(pd);
        layers[conv_idx].weights_mem = wei_mem;
        layers[conv_idx].bias_mem = bias_mem;
        layers[conv_idx].src_scale_mem = src_sc;
        layers[conv_idx].dst_scale_mem = dst_sc;
        layers[conv_idx].oc = p.out_channels;
        layers[conv_idx].oh = oh;
        layers[conv_idx].ow = ow;
    }

    void create_block_convs(int bi) {
        auto& blk = blocks[bi];
        auto& sc = block_scales[bi];

        // Determine input spatial from conv params
        auto& c1p = model.convs[blk.conv1_idx].p;
        int ic = c1p.in_channels;

        // Input spatial: depends on which layer we're in
        // layer1: 32x32, layer2: 16x16 (after stride-2 in first block), etc.
        // We can compute from the block index
        int ih, iw;
        if (bi < 2) { ih = iw = 32; }
        else if (bi < 4) { ih = iw = (bi == 2) ? 32 : 16; }
        else if (bi < 6) { ih = iw = (bi == 4) ? 16 : 8; }
        else { ih = iw = (bi == 6) ? 8 : 4; }

        // For stride-2 blocks, first conv downsamples
        auto& p1 = model.convs[blk.conv1_idx].p;
        int mid_h = (ih + 2 * p1.padding - p1.kernel_h) / p1.stride + 1;
        int mid_w = (iw + 2 * p1.padding - p1.kernel_w) / p1.stride + 1;

        // Input scale for this block
        float in_scale;
        if (bi == 0) in_scale = initial_relu_scale;
        else in_scale = block_scales[bi - 1].block_relu_out;

        // Conv1: u8 → relu → u8
        create_int8_conv(blk.conv1_idx, ic, ih, iw,
                         in_scale, sc.conv1_relu_out, 1);

        // Conv2: u8 → conv → sum(residual) + relu → u8
        int conv2_ic = model.convs[blk.conv2_idx].p.in_channels;

        if (blk.shortcut_idx >= 0) {
            // Shortcut conv: output f32 (no relu — relu is after residual add)
            create_int8_conv(blk.shortcut_idx, ic, ih, iw,
                             in_scale, sc.block_relu_out, 0);

            // Conv2 with sum: residual comes from shortcut (quantized to block_relu_out scale)
            // But shortcut is f32 — we'll handle this in forward() manually
            // So conv2 also outputs f32
            create_int8_conv(blk.conv2_idx, conv2_ic, mid_h, mid_w,
                             sc.conv1_relu_out, sc.block_relu_out, 0);
        } else {
            // Identity shortcut: residual is input x, quantized with in_scale
            create_int8_conv(blk.conv2_idx, conv2_ic, mid_h, mid_w,
                             sc.conv1_relu_out, sc.block_relu_out, 2, in_scale);
        }
    }

    void create_fc() {
        using namespace dnnl;
        using dt = memory::data_type;
        using ft = memory::format_tag;

        auto& fc = model.fc;
        int in_f = fc.in_features, out_f = fc.out_features;

        auto src_md = memory::desc({1, in_f}, dt::f32, ft::nc);
        auto wei_md = memory::desc({out_f, in_f}, dt::f32, ft::any);
        auto bias_md = memory::desc({out_f}, dt::f32, ft::x);
        auto dst_md = memory::desc({1, out_f}, dt::f32, ft::nc);

        auto pd = inner_product_forward::primitive_desc(eng, prop_kind::forward_inference,
            src_md, wei_md, bias_md, dst_md);

        fc_weights_mem = memory(pd.weights_desc(), eng);
        {
            auto user_wei = memory({{out_f, in_f}, dt::f32, ft::oi}, eng);
            write_to_dnnl_memory(fc.weight.data(), user_wei);
            reorder(user_wei, fc_weights_mem).execute(strm, user_wei, fc_weights_mem);
            strm.wait();
        }

        fc_bias_mem = memory(pd.bias_desc(), eng);
        write_to_dnnl_memory(fc.bias.data(), fc_bias_mem);

        fc_pd = pd;
        fc_prim = inner_product_forward(pd);
    }

    void load_scales(const char* path) {
        FILE* f = fopen(path, "r");
        if (!f) { fprintf(stderr, "Cannot open scales: %s\n", path); exit(1); }
        fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
        std::string json(len, '\0');
        size_t nread = fread(&json[0], 1, len, f); (void)nread;
        fclose(f);

        auto get_scale = [&](const std::string& key) -> float {
            size_t pos = json.find("\"" + key + "\"");
            if (pos == std::string::npos) { fprintf(stderr, "Missing scale: %s\n", key.c_str()); return 1.0f; }
            pos = json.find("\"scale\"", pos);
            pos = json.find(":", pos) + 1;
            return std::stof(json.substr(pos));
        };

        initial_relu_scale = get_scale("conv1_relu_out");

        // Build block structure first (needed for count)
        int num_blocks = 8;
        block_scales.resize(num_blocks);
        const char* layer_names[] = {"layer1", "layer2", "layer3", "layer4"};
        int bi = 0;
        bool has_sc[] = {false, true, true, true};
        for (int l = 0; l < 4; l++) {
            for (int b = 0; b < 2; b++) {
                char buf[64];
                snprintf(buf, sizeof(buf), "%s.%d.conv1_relu_out", layer_names[l], b);
                block_scales[bi].conv1_relu_out = get_scale(buf);
                snprintf(buf, sizeof(buf), "%s.%d.conv2_bn_out", layer_names[l], b);
                block_scales[bi].conv2_bn_out = get_scale(buf);
                if (b == 0 && has_sc[l]) {
                    snprintf(buf, sizeof(buf), "%s.%d.shortcut_out", layer_names[l], b);
                    block_scales[bi].shortcut_out = get_scale(buf);
                }
                snprintf(buf, sizeof(buf), "%s.%d.block_relu_out", layer_names[l], b);
                block_scales[bi].block_relu_out = get_scale(buf);
                bi++;
            }
        }
    }

    std::vector<float> forward(const std::vector<float>& input_data) {
        using namespace dnnl;
        using dt = memory::data_type;
        using ft = memory::format_tag;

        // ── First conv: f32 → f32 (with ReLU) → quantize to u8 ──
        auto src_mem = memory({{1, 3, 32, 32}, dt::f32, ft::nchw}, eng);
        write_to_dnnl_memory(input_data.data(), src_mem);

        auto c0_src = memory(conv0_pd.src_desc(), eng);
        if (conv0_pd.src_desc() != src_mem.get_desc()) {
            reorder(src_mem, c0_src).execute(strm, src_mem, c0_src);
        } else {
            c0_src = src_mem;
        }

        auto c0_dst_f32 = memory(conv0_pd.dst_desc(), eng);
        conv0_prim.execute(strm, {
            {DNNL_ARG_SRC, c0_src},
            {DNNL_ARG_WEIGHTS, conv0_weights_mem},
            {DNNL_ARG_BIAS, conv0_bias_mem},
            {DNNL_ARG_DST, c0_dst_f32}
        });
        strm.wait();

        // Quantize f32 → u8 via reorder with scaling
        auto u8_md = memory::desc({1, conv0_oc, conv0_oh, conv0_ow}, dt::u8, ft::any);
        // Use first int8 conv's expected src format if available
        dnnl::memory::desc target_u8_md = layers[blocks[0].conv1_idx].conv_pd.src_desc();
        auto current = memory(target_u8_md, eng);
        {
            primitive_attr rattr;
            rattr.set_scales_mask(DNNL_ARG_DST, 0);
            // dst_scale = activation scale: real = u8 * scale, so u8 = real / scale
            auto sc_mem = memory({{1}, dt::f32, ft::x}, eng);
            write_to_dnnl_memory(&initial_relu_scale, sc_mem);

            // Reorder f32 → u8 with destination scale
            auto rpd = reorder::primitive_desc(eng, c0_dst_f32.get_desc(), eng, target_u8_md, rattr);
            reorder(rpd).execute(strm, {
                {DNNL_ARG_FROM, c0_dst_f32},
                {DNNL_ARG_TO, current},
                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, sc_mem}
            });
            strm.wait();
        }

        // ── ResNet blocks ──
        for (size_t bi = 0; bi < blocks.size(); bi++) {
            auto& blk = blocks[bi];
            auto& l1 = layers[blk.conv1_idx];
            auto& l2 = layers[blk.conv2_idx];

            // Conv1: current → reorder if needed → conv → out1
            auto conv1_src = memory(l1.conv_pd.src_desc(), eng);
            if (l1.conv_pd.src_desc() != current.get_desc()) {
                reorder(current, conv1_src).execute(strm, current, conv1_src);
            } else {
                conv1_src = current;
            }

            auto out1 = memory(l1.conv_pd.dst_desc(), eng);
            l1.conv_prim.execute(strm, {
                {DNNL_ARG_SRC, conv1_src},
                {DNNL_ARG_WEIGHTS, l1.weights_mem},
                {DNNL_ARG_BIAS, l1.bias_mem},
                {DNNL_ARG_DST, out1},
                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, l1.src_scale_mem},
                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, l1.wei_scale_mem},
                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, l1.dst_scale_mem}
            });

            // Prepare residual for conv2's sum post-op
            // Conv2 dst = residual buffer (sum post-op adds conv output to it)
            auto conv2_dst = memory(l2.conv_pd.dst_desc(), eng);

            if (blk.shortcut_idx >= 0) {
                // Shortcut conv → f32 output (no relu, mode 0)
                auto& ls = layers[blk.shortcut_idx];
                auto sc_src = memory(ls.conv_pd.src_desc(), eng);
                if (ls.conv_pd.src_desc() != current.get_desc()) {
                    reorder(current, sc_src).execute(strm, current, sc_src);
                } else {
                    sc_src = current;
                }
                conv2_dst = memory(ls.conv_pd.dst_desc(), eng);  // f32
                ls.conv_prim.execute(strm, {
                    {DNNL_ARG_SRC, sc_src},
                    {DNNL_ARG_WEIGHTS, ls.weights_mem},
                    {DNNL_ARG_BIAS, ls.bias_mem},
                    {DNNL_ARG_DST, conv2_dst},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, ls.src_scale_mem},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, ls.wei_scale_mem}
                });
            } else {
                // Identity: copy current u8 activation into conv2_dst for sum post-op
                if (l2.conv_pd.dst_desc() != current.get_desc()) {
                    reorder(current, conv2_dst).execute(strm, current, conv2_dst);
                } else {
                    size_t sz = l2.conv_pd.dst_desc().get_size();
                    memcpy(conv2_dst.get_data_handle(), current.get_data_handle(), sz);
                }
            }

            // Conv2
            auto conv2_src = memory(l2.conv_pd.src_desc(), eng);
            if (l2.conv_pd.src_desc() != out1.get_desc()) {
                reorder(out1, conv2_src).execute(strm, out1, conv2_src);
            } else {
                conv2_src = out1;
            }

            if (blk.shortcut_idx >= 0) {
                // Both shortcut and conv2 output f32 — manual add+relu+requant
                auto conv2_f32 = memory(l2.conv_pd.dst_desc(), eng);
                l2.conv_prim.execute(strm, {
                    {DNNL_ARG_SRC, conv2_src},
                    {DNNL_ARG_WEIGHTS, l2.weights_mem},
                    {DNNL_ARG_BIAS, l2.bias_mem},
                    {DNNL_ARG_DST, conv2_f32},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, l2.src_scale_mem},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, l2.wei_scale_mem}
                });
                strm.wait();

                // Read both f32 outputs, add + relu + requant
                int tot = l2.oc * l2.oh * l2.ow;
                std::vector<float> c2_data(tot), sc_data(tot);

                auto c2_user = memory({{1, l2.oc, l2.oh, l2.ow}, dt::f32, ft::nchw}, eng);
                auto sc_user = memory({{1, l2.oc, l2.oh, l2.ow}, dt::f32, ft::nchw}, eng);
                reorder(conv2_f32, c2_user).execute(strm, conv2_f32, c2_user);
                reorder(conv2_dst, sc_user).execute(strm, conv2_dst, sc_user);
                strm.wait();
                read_from_dnnl_memory(c2_data.data(), c2_user);
                read_from_dnnl_memory(sc_data.data(), sc_user);

                float out_scale = block_scales[bi].block_relu_out;
                std::vector<uint8_t> u8_result(tot);
                for (int i = 0; i < tot; i++) {
                    float v = c2_data[i] + sc_data[i];
                    v = std::max(0.0f, v);  // ReLU
                    int q = (int)roundf(v / out_scale);
                    u8_result[i] = (uint8_t)std::max(0, std::min(255, q));
                }

                // Get expected format for next layer's input
                int next_conv = (bi + 1 < blocks.size()) ? blocks[bi + 1].conv1_idx : -1;
                dnnl::memory::desc target_md;
                if (next_conv >= 0) {
                    target_md = layers[next_conv].conv_pd.src_desc();
                } else {
                    target_md = memory::desc({1, l2.oc, l2.oh, l2.ow}, dt::u8, ft::nchw);
                }

                auto u8_nchw = memory({{1, l2.oc, l2.oh, l2.ow}, dt::u8, ft::nchw}, eng);
                write_to_dnnl_memory(u8_result.data(), u8_nchw);
                current = memory(target_md, eng);
                if (target_md != u8_nchw.get_desc()) {
                    reorder(u8_nchw, current).execute(strm, u8_nchw, current);
                    strm.wait();
                } else {
                    current = u8_nchw;
                }
            } else {
                // Identity shortcut: sum post-op handles it (conv2 has epilogue=2)
                l2.conv_prim.execute(strm, {
                    {DNNL_ARG_SRC, conv2_src},
                    {DNNL_ARG_WEIGHTS, l2.weights_mem},
                    {DNNL_ARG_BIAS, l2.bias_mem},
                    {DNNL_ARG_DST, conv2_dst},  // sum post-op reads+writes this
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, l2.src_scale_mem},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, l2.wei_scale_mem},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, l2.dst_scale_mem}
                });
                current = conv2_dst;
            }

        }

        // ── Avgpool (manual: dequant u8 → f32, pool) ──
        // Get last block's output info
        auto& last_blk = blocks.back();
        auto& last_l2 = layers[last_blk.conv2_idx];
        int pool_c = last_l2.oc, pool_h = last_l2.oh, pool_w = last_l2.ow;
        float last_scale = block_scales.back().block_relu_out;

        // Read u8 output
        std::vector<uint8_t> u8_out(pool_c * pool_h * pool_w);
        {
            auto user_dst = memory({{1, pool_c, pool_h, pool_w}, dt::u8, ft::nchw}, eng);
            if (current.get_desc() != user_dst.get_desc()) {
                reorder(current, user_dst).execute(strm, current, user_dst);
                strm.wait();
            } else {
                user_dst = current;
                strm.wait();
            }
            read_from_dnnl_memory(u8_out.data(), user_dst);
        }

        // Avgpool + dequant
        int spatial = pool_h * pool_w;
        std::vector<float> pooled(pool_c);
        for (int c = 0; c < pool_c; c++) {
            int32_t sum = 0;
            for (int i = 0; i < spatial; i++)
                sum += u8_out[c * spatial + i];
            pooled[c] = sum * last_scale / spatial;
        }

        // ── FC ──
        auto fc_src = memory(fc_pd.src_desc(), eng);
        write_to_dnnl_memory(pooled.data(), fc_src);
        auto fc_dst = memory(fc_pd.dst_desc(), eng);

        fc_prim.execute(strm, {
            {DNNL_ARG_SRC, fc_src},
            {DNNL_ARG_WEIGHTS, fc_weights_mem},
            {DNNL_ARG_BIAS, fc_bias_mem},
            {DNNL_ARG_DST, fc_dst}
        });
        strm.wait();

        std::vector<float> logits(model.fc.out_features);
        read_from_dnnl_memory(logits.data(), fc_dst);
        return logits;
    }
};
