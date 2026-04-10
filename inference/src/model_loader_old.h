#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

constexpr uint32_t MAGIC = 0x54524E59; // "TRNY"

enum LayerType : uint8_t {
    FP32_CONV = 0,
    TERNARY_CONV = 1,
    FC_LAYER = 2
};

struct ConvParams {
    uint32_t out_channels;
    uint32_t in_channels;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t stride;
    uint32_t padding;
};

struct FP32ConvLayer {
    ConvParams params;
    std::vector<float> weights;      // out_c * in_c * kH * kW
    std::vector<float> fused_scale;  // out_c
    std::vector<float> fused_bias;   // out_c
};

struct TernaryConvLayer {
    ConvParams params;
    std::vector<uint8_t> mask_pos;   // bitpacked
    std::vector<uint8_t> mask_neg;   // bitpacked
    std::vector<float> fused_scale;  // out_c (= bn_scale * alpha)
    std::vector<float> fused_bias;   // out_c
};

struct FCLayer {
    uint32_t out_features;
    uint32_t in_features;
    std::vector<float> weights;      // out_f * in_f
    std::vector<float> bias;         // out_f
};

struct Layer {
    LayerType type;
    FP32ConvLayer fp32_conv;
    TernaryConvLayer ternary_conv;
    FCLayer fc;
};

struct Model {
    std::vector<Layer> layers;

    bool load(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) {
            fprintf(stderr, "Cannot open %s\n", path.c_str());
            return false;
        }

        // Read header
        uint32_t magic, num_layers;
        fread(&magic, 4, 1, f);
        fread(&num_layers, 4, 1, f);

        if (magic != MAGIC) {
            fprintf(stderr, "Invalid magic: 0x%08X (expected 0x%08X)\n", magic, MAGIC);
            fclose(f);
            return false;
        }

        printf("Loading model: %u layers\n", num_layers);
        layers.resize(num_layers);

        for (uint32_t i = 0; i < num_layers; i++) {
            uint8_t layer_type, has_bn;
            ConvParams p;

            fread(&layer_type, 1, 1, f);
            fread(&p.out_channels, 4, 1, f);
            fread(&p.in_channels, 4, 1, f);
            fread(&p.kernel_h, 4, 1, f);
            fread(&p.kernel_w, 4, 1, f);
            fread(&p.stride, 4, 1, f);
            fread(&p.padding, 4, 1, f);
            fread(&has_bn, 1, 1, f);

            layers[i].type = static_cast<LayerType>(layer_type);

            if (layer_type == FP32_CONV) {
                auto& layer = layers[i].fp32_conv;
                layer.params = p;
                uint32_t wsize = p.out_channels * p.in_channels * p.kernel_h * p.kernel_w;

                layer.weights.resize(wsize);
                fread(layer.weights.data(), 4, wsize, f);

                if (has_bn) {
                    layer.fused_scale.resize(p.out_channels);
                    layer.fused_bias.resize(p.out_channels);
                    fread(layer.fused_scale.data(), 4, p.out_channels, f);
                    fread(layer.fused_bias.data(), 4, p.out_channels, f);
                }

                printf("  Layer %u: FP32 Conv %ux%ux%ux%u stride=%u pad=%u\n",
                       i, p.out_channels, p.in_channels, p.kernel_h, p.kernel_w,
                       p.stride, p.padding);

            } else if (layer_type == TERNARY_CONV) {
                auto& layer = layers[i].ternary_conv;
                layer.params = p;
                uint32_t total_weights = p.out_channels * p.in_channels * p.kernel_h * p.kernel_w;
                uint32_t packed_bytes = (total_weights + 7) / 8;

                layer.mask_pos.resize(packed_bytes);
                layer.mask_neg.resize(packed_bytes);
                fread(layer.mask_pos.data(), 1, packed_bytes, f);
                fread(layer.mask_neg.data(), 1, packed_bytes, f);

                layer.fused_scale.resize(p.out_channels);
                layer.fused_bias.resize(p.out_channels);
                fread(layer.fused_scale.data(), 4, p.out_channels, f);
                fread(layer.fused_bias.data(), 4, p.out_channels, f);

                printf("  Layer %u: Ternary Conv %ux%ux%ux%u stride=%u pad=%u (%u packed bytes)\n",
                       i, p.out_channels, p.in_channels, p.kernel_h, p.kernel_w,
                       p.stride, p.padding, packed_bytes);

            } else if (layer_type == FC_LAYER) {
                auto& layer = layers[i].fc;
                layer.out_features = p.out_channels;
                layer.in_features = p.in_channels;

                uint32_t wsize = layer.out_features * layer.in_features;
                layer.weights.resize(wsize);
                layer.bias.resize(layer.out_features);
                fread(layer.weights.data(), 4, wsize, f);
                fread(layer.bias.data(), 4, layer.out_features, f);

                printf("  Layer %u: FC %ux%u\n", i, layer.out_features, layer.in_features);
            }
        }

        fclose(f);
        printf("Model loaded successfully.\n\n");
        return true;
    }
};
