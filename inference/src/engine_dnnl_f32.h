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
// oneDNN FP32 ResNet-18 Inference Engine
//
// All convolutions run in FP32 through oneDNN's fused micro-kernels.
// BN fused into conv weights (same as our engine.h).
// Post-ops: fused ReLU and residual sum.
//
// Purpose: verify oneDNN latency without int8 scale complexity.
// ═══════════════════════════════════════════════════════════════

static void write_to_mem(const void* src, dnnl::memory& dst) {
    memcpy(dst.get_data_handle(), src, dst.get_desc().get_size());
}

static void read_from_mem(void* dst, const dnnl::memory& src) {
    memcpy(dst, src.get_data_handle(), src.get_desc().get_size());
}

struct ConvPrim {
    dnnl::convolution_forward prim;
    dnnl::convolution_forward::primitive_desc pd;
    dnnl::memory weights, bias;
    int oc, oh, ow;
    bool has_sum;  // sum post-op for residual
};

class EngineDNNL_F32 {
public:
    dnnl::engine eng;
    dnnl::stream strm;
    Model model;

    std::vector<ConvPrim> convs;
    struct Block { int conv1, conv2, shortcut; };
    std::vector<Block> blocks;

    // FC
    dnnl::inner_product_forward fc_prim;
    dnnl::inner_product_forward::primitive_desc fc_pd;
    dnnl::memory fc_wei, fc_bias;

    void init(const Model& m, const char*) {
        model = m;
        eng = dnnl::engine(dnnl::engine::kind::cpu, 0);
        strm = dnnl::stream(eng);

        int idx = 1;
        bool has_sc[] = {false, true, true, true};
        for (int l = 0; l < 4; l++)
            for (int b = 0; b < 2; b++) {
                Block bi;
                bi.conv1 = idx++; bi.conv2 = idx++;
                bi.shortcut = (b == 0 && has_sc[l]) ? idx++ : -1;
                blocks.push_back(bi);
            }

        convs.resize(model.convs.size());
        create_conv0();
        for (size_t bi = 0; bi < blocks.size(); bi++)
            create_block(bi);
        create_fc();

        printf("EngineDNNL_F32: %zu convs, %zu blocks, oneDNN FP32\n",
               model.convs.size(), blocks.size());
    }

    ConvPrim make_conv(int ci, int ic, int ih, int iw, bool fuse_relu, bool fuse_sum) {
        using namespace dnnl;
        using dt = memory::data_type;
        using ft = memory::format_tag;

        auto& fc = model.convs[ci];
        auto& p = fc.p;
        int oh = (ih + 2*p.padding - p.kernel_h) / p.stride + 1;
        int ow = (iw + 2*p.padding - p.kernel_w) / p.stride + 1;

        auto src_md = memory::desc({1, ic, ih, iw}, dt::f32, ft::any);
        auto wei_md = memory::desc({p.out_channels, ic, p.kernel_h, p.kernel_w}, dt::f32, ft::any);
        auto bias_md = memory::desc({p.out_channels}, dt::f32, ft::x);
        auto dst_md = memory::desc({1, p.out_channels, oh, ow}, dt::f32, ft::any);

        post_ops ops;
        if (fuse_sum) ops.append_sum(1.0f);
        if (fuse_relu) ops.append_eltwise(algorithm::eltwise_relu, 0.f, 0.f);
        primitive_attr attr;
        attr.set_post_ops(ops);

        auto pd = convolution_forward::primitive_desc(eng, prop_kind::forward_inference,
            algorithm::convolution_direct, src_md, wei_md, bias_md, dst_md,
            {p.stride, p.stride}, {p.padding, p.padding}, {p.padding, p.padding}, attr);

        // Prepare FP32 weights: for ternary, reconstruct alpha * ternary * bn_scale
        auto wei = memory(pd.weights_desc(), eng);
        {
            int fan_in = ic * p.kernel_h * p.kernel_w;
            std::vector<float> w_fp32(p.out_channels * fan_in);

            if (fc.is_ternary) {
                for (int o = 0; o < p.out_channels; o++) {
                    float s = fc.alpha[o] * fc.fused_scale[o];
                    for (int k = 0; k < fan_in; k++)
                        w_fp32[o * fan_in + k] = fc.weights_i8[o * fan_in + k] * s;
                }
            } else {
                // FP32 conv: weights already have BN scale fused
                w_fp32 = fc.weights_fp32;
            }

            auto user = memory({{p.out_channels, ic, p.kernel_h, p.kernel_w}, dt::f32, ft::oihw}, eng);
            write_to_mem(w_fp32.data(), user);
            reorder(user, wei).execute(strm, user, wei);
            strm.wait();
        }

        auto bias = memory(pd.bias_desc(), eng);
        write_to_mem(fc.fused_bias.data(), bias);

        ConvPrim cp;
        cp.pd = pd;
        cp.prim = convolution_forward(pd);
        cp.weights = wei;
        cp.bias = bias;
        cp.oc = p.out_channels;
        cp.oh = oh;
        cp.ow = ow;
        cp.has_sum = fuse_sum;
        return cp;
    }

    void create_conv0() {
        convs[0] = make_conv(0, 3, 32, 32, true, false);
    }

    void create_block(int bi) {
        auto& blk = blocks[bi];
        auto& c1p = model.convs[blk.conv1].p;
        int ic = c1p.in_channels;
        int ih, iw;
        if (bi < 2) { ih = iw = 32; }
        else if (bi < 4) { ih = iw = (bi == 2) ? 32 : 16; }
        else if (bi < 6) { ih = iw = (bi == 4) ? 16 : 8; }
        else { ih = iw = (bi == 6) ? 8 : 4; }

        auto& p1 = model.convs[blk.conv1].p;
        int mh = (ih + 2*p1.padding - p1.kernel_h) / p1.stride + 1;
        int mw = (iw + 2*p1.padding - p1.kernel_w) / p1.stride + 1;

        // Conv1: relu, no sum
        convs[blk.conv1] = make_conv(blk.conv1, ic, ih, iw, true, false);

        // Conv2: sum + relu (residual add fused)
        int c2ic = model.convs[blk.conv2].p.in_channels;
        convs[blk.conv2] = make_conv(blk.conv2, c2ic, mh, mw, true, true);

        // Shortcut: no relu, no sum (relu is after residual add in conv2)
        if (blk.shortcut >= 0) {
            convs[blk.shortcut] = make_conv(blk.shortcut, ic, ih, iw, false, false);
        }
    }

    void create_fc() {
        using namespace dnnl;
        using dt = memory::data_type;
        using ft = memory::format_tag;

        auto& f = model.fc;
        auto src_md = memory::desc({1, f.in_features}, dt::f32, ft::nc);
        auto wei_md = memory::desc({f.out_features, f.in_features}, dt::f32, ft::any);
        auto bias_md = memory::desc({f.out_features}, dt::f32, ft::x);
        auto dst_md = memory::desc({1, f.out_features}, dt::f32, ft::nc);

        fc_pd = inner_product_forward::primitive_desc(eng, prop_kind::forward_inference,
            src_md, wei_md, bias_md, dst_md);

        fc_wei = memory(fc_pd.weights_desc(), eng);
        {
            auto user = memory({{f.out_features, f.in_features}, dt::f32, ft::oi}, eng);
            write_to_mem(f.weight.data(), user);
            reorder(user, fc_wei).execute(strm, user, fc_wei);
            strm.wait();
        }
        fc_bias = memory(fc_pd.bias_desc(), eng);
        write_to_mem(f.bias.data(), fc_bias);
        fc_prim = inner_product_forward(fc_pd);
    }

    dnnl::memory exec_conv(int ci, dnnl::memory src, dnnl::memory* sum_src = nullptr) {
        auto& c = convs[ci];
        auto conv_src = dnnl::memory(c.pd.src_desc(), eng);
        if (c.pd.src_desc() != src.get_desc()) {
            dnnl::reorder(src, conv_src).execute(strm, src, conv_src);
        } else {
            conv_src = src;
        }

        auto dst = dnnl::memory(c.pd.dst_desc(), eng);
        if (c.has_sum && sum_src) {
            // Copy residual into dst for sum post-op
            if (c.pd.dst_desc() != sum_src->get_desc()) {
                dnnl::reorder(*sum_src, dst).execute(strm, *sum_src, dst);
            } else {
                memcpy(dst.get_data_handle(), sum_src->get_data_handle(),
                       c.pd.dst_desc().get_size());
            }
        }

        c.prim.execute(strm, {
            {DNNL_ARG_SRC, conv_src},
            {DNNL_ARG_WEIGHTS, c.weights},
            {DNNL_ARG_BIAS, c.bias},
            {DNNL_ARG_DST, dst}
        });
        return dst;
    }

    std::vector<float> forward(const std::vector<float>& input_data) {
        using namespace dnnl;
        using dt = memory::data_type;
        using ft = memory::format_tag;

        auto src = memory({{1, 3, 32, 32}, dt::f32, ft::nchw}, eng);
        write_to_mem(input_data.data(), src);

        // First conv + relu
        auto current = exec_conv(0, src);

        // ResNet blocks
        for (size_t bi = 0; bi < blocks.size(); bi++) {
            auto& blk = blocks[bi];

            auto out1 = exec_conv(blk.conv1, current);

            if (blk.shortcut >= 0) {
                // Shortcut (no relu)
                auto shortcut = exec_conv(blk.shortcut, current);
                // Conv2 with sum(shortcut) + relu
                current = exec_conv(blk.conv2, out1, &shortcut);
            } else {
                // Conv2 with sum(identity) + relu
                current = exec_conv(blk.conv2, out1, &current);
            }
        }

        strm.wait();

        // Avgpool + FC
        auto& last = convs[blocks.back().conv2];
        int pc = last.oc, ph = last.oh, pw = last.ow;
        std::vector<float> pool_in(pc * ph * pw);
        {
            auto user = memory({{1, pc, ph, pw}, dt::f32, ft::nchw}, eng);
            if (current.get_desc() != user.get_desc()) {
                reorder(current, user).execute(strm, current, user);
                strm.wait();
            } else {
                user = current;
            }
            read_from_mem(pool_in.data(), user);
        }

        int spatial = ph * pw;
        std::vector<float> pooled(pc);
        for (int c = 0; c < pc; c++) {
            float s = 0;
            for (int i = 0; i < spatial; i++) s += pool_in[c * spatial + i];
            pooled[c] = s / spatial;
        }

        auto fc_src = memory(fc_pd.src_desc(), eng);
        write_to_mem(pooled.data(), fc_src);
        auto fc_dst = memory(fc_pd.dst_desc(), eng);
        fc_prim.execute(strm, {
            {DNNL_ARG_SRC, fc_src},
            {DNNL_ARG_WEIGHTS, fc_wei},
            {DNNL_ARG_BIAS, fc_bias},
            {DNNL_ARG_DST, fc_dst}
        });
        strm.wait();

        std::vector<float> logits(model.fc.out_features);
        read_from_mem(logits.data(), fc_dst);
        return logits;
    }
};
