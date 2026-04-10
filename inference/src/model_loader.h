#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

// ─── Binary format structs ──────────────────────────────────

struct ConvParams {
    uint16_t out_channels, in_channels, kernel_h, kernel_w;
    uint16_t stride, padding;
};

struct FP32Conv {
    ConvParams p;
    std::string name;
    std::vector<float> weights;  // OC * IC * kH * kW
};

struct TernaryConv {
    ConvParams p;
    std::string name;
    std::vector<float> alpha;         // per OC
    std::vector<uint8_t> packed_w;    // I2_S packed
    std::vector<int8_t> weights_i8;   // unpacked to {-1, 0, +1} for VNNI
};

struct BatchNorm {
    std::string name;
    uint16_t num_features;
    std::vector<float> weight, bias, running_mean, running_var;
    float eps;
    float act_scale;  // static activation scale
};

struct FC {
    std::string name;
    uint16_t in_features, out_features;
    std::vector<float> weight, bias;
};

// Fused conv+BN parameters for inference
struct FusedConv {
    ConvParams p;
    std::string name;
    bool is_ternary;

    // FP32 weights (for first conv / shortcuts)
    std::vector<float> weights_fp32;

    // Ternary weights as INT8 {-1, 0, +1}
    std::vector<int8_t> weights_i8;
    std::vector<float> alpha;  // per OC

    // Fused BN: out = fused_scale * conv_out + fused_bias
    std::vector<float> fused_scale;
    std::vector<float> fused_bias;

    float act_scale;  // static quantization scale for output activations
};

struct Model {
    std::vector<FusedConv> convs;
    FC fc;

    // Layer execution order for ResNet-18:
    // conv0: first FP32 conv + BN
    // Then blocks: each block has conv1+BN, conv2+BN, optional shortcut conv+BN
    // Total: 1 + 8*2 + 3 shortcuts = 20 convs (but we'll just store them in order)
};

// ─── I2_S unpacking ─────────────────────────────────────────

static void unpack_i2s(const uint8_t* packed, int num_weights, int8_t* out) {
    // 4 weights per byte, MSB first
    // +1 -> 0b01, 0 -> 0b00, -1 -> 0b11
    int byte_idx = 0;
    for (int i = 0; i < num_weights; i += 4) {
        uint8_t byte = packed[byte_idx++];
        for (int j = 0; j < 4 && (i + j) < num_weights; j++) {
            uint8_t bits = (byte >> (6 - 2 * j)) & 0x03;
            if (bits == 0x01) out[i + j] = 1;
            else if (bits == 0x03) out[i + j] = -1;
            else out[i + j] = 0;
        }
    }
}

// ─── Fuse BN into conv weights ──────────────────────────────

static void fuse_bn(FusedConv& conv, const BatchNorm& bn) {
    int oc = conv.p.out_channels;
    conv.fused_scale.resize(oc);
    conv.fused_bias.resize(oc);
    conv.act_scale = bn.act_scale;

    for (int i = 0; i < oc; i++) {
        float std_inv = 1.0f / sqrtf(bn.running_var[i] + bn.eps);
        float scale = bn.weight[i] * std_inv;
        float bias = bn.bias[i] - bn.running_mean[i] * scale;

        conv.fused_scale[i] = scale;
        conv.fused_bias[i] = bias;

        // For FP32 convs, fuse scale into weights directly
        if (!conv.is_ternary) {
            int fan_in = conv.p.in_channels * conv.p.kernel_h * conv.p.kernel_w;
            float* w = conv.weights_fp32.data() + i * fan_in;
            for (int j = 0; j < fan_in; j++) {
                w[j] *= scale;
            }
            conv.fused_scale[i] = 1.0f;  // already folded into weights
            conv.fused_bias[i] = bias;
        }
    }
}

// ─── Model loading ──────────────────────────────────────────

static Model load_model(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open model: %s\n", path);
        exit(1);
    }

    uint32_t magic, version, num_layers;
    fread(&magic, 4, 1, f);
    fread(&version, 4, 1, f);
    fread(&num_layers, 4, 1, f);

    if (magic != 0x54524E59) {
        fprintf(stderr, "Bad magic: 0x%08X\n", magic);
        exit(1);
    }

    Model model;

    // Temporary storage: we pair conv + BN layers
    FusedConv* pending_conv = nullptr;

    for (uint32_t i = 0; i < num_layers; i++) {
        uint8_t layer_type;
        fread(&layer_type, 1, 1, f);

        uint16_t name_len;
        fread(&name_len, 2, 1, f);
        std::string name(name_len, '\0');
        fread(&name[0], 1, name_len, f);

        if (layer_type == 0) {
            // FP32 Conv
            FusedConv conv;
            conv.name = name;
            conv.is_ternary = false;
            fread(&conv.p, sizeof(ConvParams), 1, f);
            int wsize = conv.p.out_channels * conv.p.in_channels * conv.p.kernel_h * conv.p.kernel_w;
            conv.weights_fp32.resize(wsize);
            fread(conv.weights_fp32.data(), 4, wsize, f);
            model.convs.push_back(std::move(conv));
            pending_conv = &model.convs.back();

        } else if (layer_type == 1) {
            // Ternary Conv
            FusedConv conv;
            conv.name = name;
            conv.is_ternary = true;
            fread(&conv.p, sizeof(ConvParams), 1, f);

            int oc = conv.p.out_channels;
            conv.alpha.resize(oc);
            fread(conv.alpha.data(), 4, oc, f);

            uint32_t packed_size;
            fread(&packed_size, 4, 1, f);
            std::vector<uint8_t> packed(packed_size);
            fread(packed.data(), 1, packed_size, f);

            // Unpack to INT8 {-1, 0, +1}
            int num_weights = oc * conv.p.in_channels * conv.p.kernel_h * conv.p.kernel_w;
            conv.weights_i8.resize(num_weights);
            unpack_i2s(packed.data(), num_weights, conv.weights_i8.data());

            model.convs.push_back(std::move(conv));
            pending_conv = &model.convs.back();

        } else if (layer_type == 2) {
            // BatchNorm
            BatchNorm bn;
            bn.name = name;
            fread(&bn.num_features, 2, 1, f);
            int nf = bn.num_features;
            bn.weight.resize(nf); fread(bn.weight.data(), 4, nf, f);
            bn.bias.resize(nf);   fread(bn.bias.data(), 4, nf, f);
            bn.running_mean.resize(nf); fread(bn.running_mean.data(), 4, nf, f);
            bn.running_var.resize(nf);  fread(bn.running_var.data(), 4, nf, f);
            fread(&bn.eps, 4, 1, f);
            fread(&bn.act_scale, 4, 1, f);

            // Fuse BN into the pending conv
            if (pending_conv) {
                fuse_bn(*pending_conv, bn);
                pending_conv = nullptr;
            }

        } else if (layer_type == 3) {
            // FC
            fread(&model.fc.in_features, 2, 1, f);
            fread(&model.fc.out_features, 2, 1, f);
            int wsize = model.fc.in_features * model.fc.out_features;
            model.fc.weight.resize(wsize);
            fread(model.fc.weight.data(), 4, wsize, f);
            model.fc.bias.resize(model.fc.out_features);
            fread(model.fc.bias.data(), 4, model.fc.out_features, f);
            model.fc.name = name;

        } else if (layer_type == 4) {
            // Avgpool — nothing to load, handled in engine
        }
    }

    fclose(f);
    printf("Loaded model: %zu conv layers, FC %d->%d\n",
           model.convs.size(), model.fc.in_features, model.fc.out_features);
    return model;
}
