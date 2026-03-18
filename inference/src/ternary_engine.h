#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h> // AVX2
#include "model_loader.h"

// ─── Tensor: simple NCHW buffer ──────────────────────────────
struct Tensor {
    std::vector<float> data;
    int n, c, h, w;

    Tensor() : n(0), c(0), h(0), w(0) {}
    Tensor(int n, int c, int h, int w)
        : data(n * c * h * w, 0.0f), n(n), c(c), h(h), w(w) {}

    float& at(int ni, int ci, int hi, int wi) {
        return data[((ni * c + ci) * h + hi) * w + wi];
    }
    const float& at(int ni, int ci, int hi, int wi) const {
        return data[((ni * c + ci) * h + hi) * w + wi];
    }
    float* channel_ptr(int ni, int ci) {
        return data.data() + ((ni * c + ci) * h) * w;
    }
    const float* channel_ptr(int ni, int ci) const {
        return data.data() + ((ni * c + ci) * h) * w;
    }
    int total() const { return n * c * h * w; }
};

// ─── Im2col: unroll input patches into columns ──────────────
// Output: col_data of shape [in_c * kH * kW, out_h * out_w]
static void im2col(const Tensor& input, int ni,
                   int kH, int kW, int stride, int padding,
                   int out_h, int out_w,
                   std::vector<float>& col_data) {
    int in_c = input.c;
    int in_h = input.h;
    int in_w = input.w;
    int col_rows = in_c * kH * kW;
    int col_cols = out_h * out_w;

    col_data.resize(col_rows * col_cols);

    int idx = 0;
    for (int ic = 0; ic < in_c; ic++) {
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                for (int oh = 0; oh < out_h; oh++) {
                    for (int ow = 0; ow < out_w; ow++) {
                        int ih = oh * stride - padding + kh;
                        int iw = ow * stride - padding + kw;
                        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            col_data[idx] = input.at(ni, ic, ih, iw);
                        } else {
                            col_data[idx] = 0.0f; // zero padding
                        }
                        idx++;
                    }
                }
            }
        }
    }
}

// ─── FP32 Convolution (im2col + GEMM with AVX2) ─────────────
static Tensor conv2d_fp32(const Tensor& input, const FP32ConvLayer& layer) {
    auto& p = layer.params;
    int out_h = (input.h + 2 * p.padding - p.kernel_h) / p.stride + 1;
    int out_w = (input.w + 2 * p.padding - p.kernel_w) / p.stride + 1;

    Tensor output(input.n, p.out_channels, out_h, out_w);

    int col_rows = p.in_channels * p.kernel_h * p.kernel_w;
    int col_cols = out_h * out_w;
    std::vector<float> col_data;

    for (int ni = 0; ni < input.n; ni++) {
        im2col(input, ni, p.kernel_h, p.kernel_w, p.stride, p.padding,
               out_h, out_w, col_data);

        // GEMM: output[oc, :] = weights[oc, :] · col_data
        for (uint32_t oc = 0; oc < p.out_channels; oc++) {
            const float* w_row = layer.weights.data() + oc * col_rows;
            float* out_row = output.channel_ptr(ni, oc);

            for (int j = 0; j < col_cols; j++) {
                __m256 sum_vec = _mm256_setzero_ps();
                int k = 0;

                // AVX2: process 8 elements at a time
                for (; k + 7 < col_rows; k += 8) {
                    __m256 w = _mm256_loadu_ps(w_row + k);
                    __m256 c = _mm256_loadu_ps(col_data.data() + k * col_cols + j);
                    // Note: col_data is row-major [col_rows, col_cols]
                    // For proper SIMD, we'd want col-major access
                    // Falling back to scalar for correctness with row-major layout
                    break;
                }

                // Scalar fallback (correct for row-major col_data)
                float sum = 0.0f;
                for (k = 0; k < col_rows; k++) {
                    sum += w_row[k] * col_data[k * col_cols + j];
                }

                // Apply fused BN: scale * sum + bias
                if (!layer.fused_scale.empty()) {
                    sum = layer.fused_scale[oc] * sum + layer.fused_bias[oc];
                }

                out_row[j] = sum;
            }
        }
    }
    return output;
}

// ─── Ternary Convolution (im2col + SIMD bitmask) ────────────
// This is the hot path — where the speedup comes from.
// Instead of multiply-accumulate, we use bitmasks to add/subtract.
static Tensor conv2d_ternary(const Tensor& input, const TernaryConvLayer& layer) {
    auto& p = layer.params;
    int out_h = (input.h + 2 * p.padding - p.kernel_h) / p.stride + 1;
    int out_w = (input.w + 2 * p.padding - p.kernel_w) / p.stride + 1;

    Tensor output(input.n, p.out_channels, out_h, out_w);

    int col_rows = p.in_channels * p.kernel_h * p.kernel_w;
    int col_cols = out_h * out_w;
    std::vector<float> col_data;

    // Weights per output channel
    int weights_per_oc = p.in_channels * p.kernel_h * p.kernel_w;

    for (int ni = 0; ni < input.n; ni++) {
        im2col(input, ni, p.kernel_h, p.kernel_w, p.stride, p.padding,
               out_h, out_w, col_data);

        for (uint32_t oc = 0; oc < p.out_channels; oc++) {
            float* out_row = output.channel_ptr(ni, oc);

            // Bitmask offset for this output channel
            int bit_offset = oc * weights_per_oc;

            for (int j = 0; j < col_cols; j++) {
                float sum = 0.0f;

                int k = 0;

#ifdef __AVX2__
                // AVX2 SIMD path: process 8 input elements at a time
                __m256 sum_vec = _mm256_setzero_ps();

                for (; k + 7 < col_rows; k += 8) {
                    // Load 8 input values from the column
                    // col_data layout: [col_rows][col_cols], we want col_data[k..k+7][j]
                    // These are strided — need to gather
                    float vals[8];
                    for (int m = 0; m < 8; m++) {
                        vals[m] = col_data[(k + m) * col_cols + j];
                    }
                    __m256 input_vec = _mm256_loadu_ps(vals);

                    // Check bitmasks for these 8 weights
                    float pos_mask[8], neg_mask[8];
                    for (int m = 0; m < 8; m++) {
                        int bit_idx = bit_offset + k + m;
                        int byte_idx = bit_idx / 8;
                        int bit_pos = bit_idx % 8;
                        int is_pos = (layer.mask_pos[byte_idx] >> bit_pos) & 1;
                        int is_neg = (layer.mask_neg[byte_idx] >> bit_pos) & 1;
                        pos_mask[m] = is_pos ? 1.0f : 0.0f;
                        neg_mask[m] = is_neg ? 1.0f : 0.0f;
                    }

                    __m256 pos_vec = _mm256_loadu_ps(pos_mask);
                    __m256 neg_vec = _mm256_loadu_ps(neg_mask);

                    // sum += input * pos_mask - input * neg_mask
                    // = input * (pos_mask - neg_mask)
                    __m256 sign_vec = _mm256_sub_ps(pos_vec, neg_vec);
                    sum_vec = _mm256_fmadd_ps(input_vec, sign_vec, sum_vec);
                }

                // Horizontal sum of AVX2 register
                __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
                __m128 lo = _mm256_castps256_ps128(sum_vec);
                __m128 sum128 = _mm_add_ps(lo, hi);
                sum128 = _mm_hadd_ps(sum128, sum128);
                sum128 = _mm_hadd_ps(sum128, sum128);
                sum = _mm_cvtss_f32(sum128);
#endif

                // Scalar tail (remaining elements)
                for (; k < col_rows; k++) {
                    int bit_idx = bit_offset + k;
                    int byte_idx = bit_idx / 8;
                    int bit_pos = bit_idx % 8;
                    int is_pos = (layer.mask_pos[byte_idx] >> bit_pos) & 1;
                    int is_neg = (layer.mask_neg[byte_idx] >> bit_pos) & 1;

                    float val = col_data[k * col_cols + j];
                    if (is_pos) sum += val;
                    if (is_neg) sum -= val;
                }

                // Apply fused scale and bias: output = scale * sum + bias
                out_row[j] = layer.fused_scale[oc] * sum + layer.fused_bias[oc];
            }
        }
    }
    return output;
}

// ─── ReLU (in-place) ─────────────────────────────────────────
static void relu_inplace(Tensor& t) {
    int size = t.total();
    int i = 0;

#ifdef __AVX2__
    __m256 zero = _mm256_setzero_ps();
    for (; i + 7 < size; i += 8) {
        __m256 v = _mm256_loadu_ps(t.data.data() + i);
        v = _mm256_max_ps(v, zero);
        _mm256_storeu_ps(t.data.data() + i, v);
    }
#endif

    for (; i < size; i++) {
        if (t.data[i] < 0.0f) t.data[i] = 0.0f;
    }
}

// ─── Element-wise Add (for residual connections) ─────────────
static void add_inplace(Tensor& a, const Tensor& b) {
    int size = a.total();
    int i = 0;

#ifdef __AVX2__
    for (; i + 7 < size; i += 8) {
        __m256 va = _mm256_loadu_ps(a.data.data() + i);
        __m256 vb = _mm256_loadu_ps(b.data.data() + i);
        _mm256_storeu_ps(a.data.data() + i, _mm256_add_ps(va, vb));
    }
#endif

    for (; i < size; i++) {
        a.data[i] += b.data[i];
    }
}

// ─── Global Average Pooling ──────────────────────────────────
static Tensor global_avg_pool(const Tensor& input) {
    Tensor output(input.n, input.c, 1, 1);
    int spatial = input.h * input.w;
    float inv_spatial = 1.0f / spatial;

    for (int ni = 0; ni < input.n; ni++) {
        for (int ci = 0; ci < input.c; ci++) {
            const float* ptr = input.channel_ptr(ni, ci);
            float sum = 0.0f;

            int i = 0;
#ifdef __AVX2__
            __m256 sum_vec = _mm256_setzero_ps();
            for (; i + 7 < spatial; i += 8) {
                sum_vec = _mm256_add_ps(sum_vec, _mm256_loadu_ps(ptr + i));
            }
            __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
            __m128 lo = _mm256_castps256_ps128(sum_vec);
            __m128 s = _mm_add_ps(lo, hi);
            s = _mm_hadd_ps(s, s);
            s = _mm_hadd_ps(s, s);
            sum = _mm_cvtss_f32(s);
#endif
            for (; i < spatial; i++) {
                sum += ptr[i];
            }

            output.at(ni, ci, 0, 0) = sum * inv_spatial;
        }
    }
    return output;
}

// ─── Fully Connected Layer ───────────────────────────────────
static Tensor fc_forward(const Tensor& input, const FCLayer& layer) {
    // Input should be (N, in_features, 1, 1) or flattened
    int batch = input.n;
    Tensor output(batch, layer.out_features, 1, 1);

    for (int ni = 0; ni < batch; ni++) {
        const float* in_ptr = input.data.data() + ni * layer.in_features;

        for (uint32_t oc = 0; oc < layer.out_features; oc++) {
            const float* w_row = layer.weights.data() + oc * layer.in_features;
            float sum = layer.bias[oc];

            int k = 0;
#ifdef __AVX2__
            __m256 sum_vec = _mm256_setzero_ps();
            for (; k + 7 < (int)layer.in_features; k += 8) {
                __m256 w = _mm256_loadu_ps(w_row + k);
                __m256 x = _mm256_loadu_ps(in_ptr + k);
                sum_vec = _mm256_fmadd_ps(w, x, sum_vec);
            }
            __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
            __m128 lo = _mm256_castps256_ps128(sum_vec);
            __m128 s = _mm_add_ps(lo, hi);
            s = _mm_hadd_ps(s, s);
            s = _mm_hadd_ps(s, s);
            sum += _mm_cvtss_f32(s);
#endif
            for (; k < (int)layer.in_features; k++) {
                sum += w_row[k] * in_ptr[k];
            }

            output.at(ni, oc, 0, 0) = sum;
        }
    }
    return output;
}

// ─── Full ResNet-18 Inference ────────────────────────────────
// Hardcoded ResNet-18 structure since we know the architecture.
// This is cleaner than a generic graph executor for this assignment.
class TernaryResNet18 {
public:
    Model model;

    bool load(const std::string& path) {
        return model.load(path);
    }

    // Returns logits (N, 10, 1, 1)
    Tensor forward(const Tensor& input) {
        int idx = 0;

        // Layer 0: conv1 (FP32) + BN + ReLU
        Tensor x = conv2d_fp32(input, model.layers[idx++].fp32_conv);
        relu_inplace(x);

        // Residual blocks
        // layer1: 2 blocks, no shortcut (same dimensions)
        x = basic_block(x, idx, false); // block 0
        x = basic_block(x, idx, false); // block 1

        // layer2: 2 blocks, first has shortcut (dimensions change)
        x = basic_block(x, idx, true);  // block 0 (with shortcut)
        x = basic_block(x, idx, false); // block 1

        // layer3: 2 blocks, first has shortcut
        x = basic_block(x, idx, true);
        x = basic_block(x, idx, false);

        // layer4: 2 blocks, first has shortcut
        x = basic_block(x, idx, true);
        x = basic_block(x, idx, false);

        // Global average pool
        x = global_avg_pool(x);

        // FC layer
        x = fc_forward(x, model.layers[idx++].fc);

        return x;
    }

private:
    Tensor basic_block(Tensor& x, int& idx, bool has_shortcut) {
        Tensor identity = x; // copy for residual

        // Conv1 (ternary) + BN + ReLU
        Tensor out = conv2d_ternary(x, model.layers[idx++].ternary_conv);
        relu_inplace(out);

        // Conv2 (ternary) + BN (no ReLU yet)
        out = conv2d_ternary(out, model.layers[idx++].ternary_conv);

        // Shortcut
        if (has_shortcut) {
            identity = conv2d_fp32(identity, model.layers[idx++].fp32_conv);
        }

        // Residual add + ReLU
        add_inplace(out, identity);
        relu_inplace(out);

        return out;
    }
};
