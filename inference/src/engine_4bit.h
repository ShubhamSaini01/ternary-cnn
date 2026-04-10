#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>
#include <omp.h>
#include "model_loader.h"

// ═══════════════════════════════════════════════════════════════
// 4-bit Ternary ResNet-18 Inference Engine
//
// Ternary weights {-1, 0, +1} × 4-bit activations [0, 15]
// Uses _mm256_sign_epi32 + _mm256_add_epi32 instead of dpbusd.
// No spatial-packed repack — standard im2col layout feeds GEMM.
// Activations stored as uint8 with values clamped to [0, 15].
// ═══════════════════════════════════════════════════════════════

struct Tensor {
    std::vector<float> data;
    int n, c, h, w;
    Tensor() : n(0), c(0), h(0), w(0) {}
    Tensor(int n, int c, int h, int w) : data(n * c * h * w, 0.0f), n(n), c(c), h(h), w(w) {}
    float* ptr(int ni, int ci) { return data.data() + ((ni * c + ci) * h) * w; }
    const float* ptr(int ni, int ci) const { return data.data() + ((ni * c + ci) * h) * w; }
    int total() const { return n * c * h * w; }
};

struct TensorU8 {
    std::vector<uint8_t> data;
    int n, c, h, w;
    float scale;  // real_value = uint8_value * scale
    TensorU8() : n(0), c(0), h(0), w(0), scale(1.0f) {}
    TensorU8(int n, int c, int h, int w, float s) : data(n * c * h * w, 0), n(n), c(c), h(h), w(w), scale(s) {}
    uint8_t* ptr(int ni, int ci) { return data.data() + ((ni * c + ci) * h) * w; }
    const uint8_t* ptr(int ni, int ci) const { return data.data() + ((ni * c + ci) * h) * w; }
    int total() const { return n * c * h * w; }
};

// ─── Quantize FP32 → uint8 [0, 15] (4-bit range) ──────────────

static TensorU8 quantize_to_u8_4bit(const Tensor& t, float out_scale) {
    TensorU8 out(t.n, t.c, t.h, t.w, out_scale);
    float inv = 1.0f / out_scale;
    int tot = t.total(), i = 0;
    __m256 sv = _mm256_set1_ps(inv);
    __m256i lo = _mm256_setzero_si256(), hi = _mm256_set1_epi32(15);
    for (; i + 7 < tot; i += 8) {
        __m256i i32 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(t.data.data() + i), sv));
        i32 = _mm256_min_epi32(_mm256_max_epi32(i32, lo), hi);
        __m256i p16 = _mm256_packus_epi32(i32, i32);
        p16 = _mm256_permute4x64_epi64(p16, 0xD8);
        __m128i p8 = _mm_packus_epi16(_mm256_castsi256_si128(p16), _mm256_castsi256_si128(p16));
        _mm_storel_epi64((__m128i*)(out.data.data() + i), p8);
    }
    for (; i < tot; i++) {
        int v = (int)roundf(t.data[i] * inv);
        out.data[i] = (uint8_t)std::max(0, std::min(15, v));
    }
    return out;
}

// ─── im2col on uint8 data ───────────────────────────────────

static void im2col_u8_generic_4b(const TensorU8& input, int ni,
                                  int kH, int kW, int stride, int padding,
                                  int out_h, int out_w, uint8_t* col) {
    int ic = input.c, ih = input.h, iw = input.w;
    int idx = 0;
    for (int c = 0; c < ic; c++) {
        const uint8_t* ch = input.ptr(ni, c);
        for (int kh = 0; kh < kH; kh++)
            for (int kw = 0; kw < kW; kw++)
                for (int oh = 0; oh < out_h; oh++) {
                    int row = oh * stride - padding + kh;
                    for (int ow = 0; ow < out_w; ow++) {
                        int cp = ow * stride - padding + kw;
                        col[idx++] = (row >= 0 && row < ih && cp >= 0 && cp < iw)
                                     ? ch[row * iw + cp] : 0;
                    }
                }
    }
}

// Specialized 3×3 stride-1 pad-1 im2col
static void im2col_u8_3x3s1p1_4b(const TensorU8& input, int ni,
                                   int out_h, int out_w, uint8_t* col) {
    int ic = input.c, ih = input.h, iw = input.w;
    int spatial = out_h * out_w;

    for (int c = 0; c < ic; c++) {
        const uint8_t* ch = input.ptr(ni, c);
        for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
                uint8_t* dst = col;
                for (int oh_i = 0; oh_i < out_h; oh_i++) {
                    int ih_i = oh_i - 1 + kh;
                    if (ih_i < 0 || ih_i >= ih) {
                        memset(dst, 0, out_w);
                    } else {
                        const uint8_t* src_row = ch + ih_i * iw;
                        if (kw == 0) {
                            dst[0] = 0;
                            memcpy(dst + 1, src_row, iw - 1);
                        } else if (kw == 1) {
                            memcpy(dst, src_row, iw);
                        } else {
                            memcpy(dst, src_row + 1, iw - 1);
                            dst[iw - 1] = 0;
                        }
                    }
                    dst += out_w;
                }
                col += spatial;
            }
        }
    }
}

static void im2col_u8_4b(const TensorU8& input, int ni,
                          int kH, int kW, int stride, int padding,
                          int out_h, int out_w, uint8_t* col) {
    if (kH == 3 && kW == 3 && stride == 1 && padding == 1) {
        im2col_u8_3x3s1p1_4b(input, ni, out_h, out_w, col);
    } else {
        im2col_u8_generic_4b(input, ni, kH, kW, stride, padding, out_h, out_w, col);
    }
}

// ─── FP32 im2col ───────────────────────────────────────────────

static void im2col_fp32_4b(const Tensor& input, int ni,
                            int kH, int kW, int stride, int padding,
                            int out_h, int out_w, float* col) {
    int ic = input.c, ih = input.h, iw = input.w;
    int idx = 0;
    for (int c = 0; c < ic; c++) {
        const float* ch = input.ptr(ni, c);
        for (int kh = 0; kh < kH; kh++)
            for (int kw = 0; kw < kW; kw++)
                for (int oh = 0; oh < out_h; oh++) {
                    int row = oh * stride - padding + kh;
                    for (int ow = 0; ow < out_w; ow++) {
                        int cp = ow * stride - padding + kw;
                        col[idx++] = (row >= 0 && row < ih && cp >= 0 && cp < iw)
                                     ? ch[row * iw + cp] : 0.0f;
                    }
                }
    }
}

// ─── FP32 Conv (first conv) ────────────────────────────────────

static Tensor conv2d_fp32_4b(const Tensor& input, const FusedConv& layer) {
    auto& p = layer.p;
    int oh = (input.h + 2 * p.padding - p.kernel_h) / p.stride + 1;
    int ow = (input.w + 2 * p.padding - p.kernel_w) / p.stride + 1;
    Tensor output(input.n, p.out_channels, oh, ow);
    int col_rows = p.in_channels * p.kernel_h * p.kernel_w;
    int col_cols = oh * ow;
    std::vector<float> col(col_rows * col_cols);

    for (int ni = 0; ni < input.n; ni++) {
        im2col_fp32_4b(input, ni, p.kernel_h, p.kernel_w, p.stride, p.padding, oh, ow, col.data());
        for (int oc = 0; oc < p.out_channels; oc++) {
            float* out = output.ptr(ni, oc);
            const float* w = layer.weights_fp32.data() + oc * col_rows;
            float bias = layer.fused_bias[oc];
            memset(out, 0, col_cols * sizeof(float));
            for (int k = 0; k < col_rows; k++) {
                const float* src = col.data() + k * col_cols;
                __m256 wv = _mm256_set1_ps(w[k]);
                int j = 0;
                for (; j + 7 < col_cols; j += 8)
                    _mm256_storeu_ps(out + j, _mm256_fmadd_ps(wv, _mm256_loadu_ps(src + j), _mm256_loadu_ps(out + j)));
                for (; j < col_cols; j++) out[j] += w[k] * src[j];
            }
            __m256 bv = _mm256_set1_ps(bias);
            int j = 0;
            for (; j + 7 < col_cols; j += 8)
                _mm256_storeu_ps(out + j, _mm256_add_ps(_mm256_loadu_ps(out + j), bv));
            for (; j < col_cols; j++) out[j] += bias;
        }
    }
    return output;
}

// FP32 conv from uint8 input (for shortcuts)
static Tensor conv2d_fp32_from_u8_4b(const TensorU8& input, const FusedConv& layer) {
    auto& p = layer.p;
    int oh = (input.h + 2 * p.padding - p.kernel_h) / p.stride + 1;
    int ow = (input.w + 2 * p.padding - p.kernel_w) / p.stride + 1;
    Tensor output(input.n, p.out_channels, oh, ow);
    int col_rows = p.in_channels * p.kernel_h * p.kernel_w;
    int col_cols = oh * ow;

    std::vector<float> col(col_rows * col_cols);
    std::vector<uint8_t> col_u8(col_rows * col_cols);
    for (int ni = 0; ni < input.n; ni++) {
        im2col_u8_4b(input, ni, p.kernel_h, p.kernel_w, p.stride, p.padding, oh, ow, col_u8.data());
        float s = input.scale;
        for (int i = 0; i < col_rows * col_cols; i++)
            col[i] = col_u8[i] * s;

        for (int oc = 0; oc < p.out_channels; oc++) {
            float* out = output.ptr(ni, oc);
            const float* w = layer.weights_fp32.data() + oc * col_rows;
            float bias = layer.fused_bias[oc];
            memset(out, 0, col_cols * sizeof(float));
            for (int k = 0; k < col_rows; k++) {
                const float* src = col.data() + k * col_cols;
                __m256 wv = _mm256_set1_ps(w[k]);
                int j = 0;
                for (; j + 7 < col_cols; j += 8)
                    _mm256_storeu_ps(out + j, _mm256_fmadd_ps(wv, _mm256_loadu_ps(src + j), _mm256_loadu_ps(out + j)));
                for (; j < col_cols; j++) out[j] += w[k] * src[j];
            }
            __m256 bv = _mm256_set1_ps(bias);
            int j = 0;
            for (; j + 7 < col_cols; j += 8)
                _mm256_storeu_ps(out + j, _mm256_add_ps(_mm256_loadu_ps(out + j), bv));
            for (; j < col_cols; j++) out[j] += bias;
        }
    }
    return output;
}

// ─── Add/Sub Kernel Data ───────────────────────────────────────

struct AddSubKernelData {
    int col_rows;
    // Raw ternary weights: [OC][col_rows], flat int8 {-1, 0, +1}
    // No packing needed — sign_epi32 handles them directly
    std::vector<int8_t> weights_i8;

    // Precomputed per-OC constants
    std::vector<float> oc_scale;  // alpha[oc] * fused_bn_scale[oc]
    std::vector<float> oc_bias;   // fused_bn_bias[oc]
};

static void prepare_addsub_kernel(const FusedConv& layer, AddSubKernelData& kd) {
    int noc = layer.p.out_channels;
    int ic = layer.p.in_channels, kH = layer.p.kernel_h, kW = layer.p.kernel_w;
    kd.col_rows = ic * kH * kW;

    // Copy ternary weights directly — no repack!
    kd.weights_i8.resize(noc * kd.col_rows);
    memcpy(kd.weights_i8.data(), layer.weights_i8.data(), noc * kd.col_rows);

    kd.oc_scale.resize(noc);
    kd.oc_bias.resize(noc);
    for (int o = 0; o < noc; o++) {
        kd.oc_scale[o] = layer.alpha[o] * layer.fused_scale[o];
        kd.oc_bias[o] = layer.fused_bias[o];
    }
}

// ─── Ternary Conv: sign_epi32 + add_epi32 GEMM ────────────────
//
// Epilogue modes:
//   0: output FP32 (out_fp32)
//   1: fused ReLU + requant → uint8 [0,15] (out_u8) — for conv1
//   2: fused residual + ReLU + requant → uint8 [0,15] (out_u8) — for conv2

static void conv2d_ternary_addsub(const TensorU8& input, const FusedConv& layer,
                                   const AddSubKernelData& kd,
                                   uint8_t* col_u8,  // preallocated workspace
                                   int epilogue_mode,
                                   Tensor* out_fp32,
                                   TensorU8* out_u8, float out_u8_scale,
                                   const TensorU8* residual_u8 = nullptr,
                                   const Tensor* residual_fp32 = nullptr,
                                   float residual_scale = 0.0f) {
    auto& p = layer.p;
    int oh = (input.h + 2*p.padding - p.kernel_h) / p.stride + 1;
    int ow = (input.w + 2*p.padding - p.kernel_w) / p.stride + 1;

    if (epilogue_mode == 0) *out_fp32 = Tensor(input.n, p.out_channels, oh, ow);
    else *out_u8 = TensorU8(input.n, p.out_channels, oh, ow, out_u8_scale);

    int col_rows = kd.col_rows;
    int col_cols = oh * ow;

    const int OC_TILE = 16;
    float act_scale = input.scale;

    for (int ni = 0; ni < input.n; ni++) {
        // Standard im2col — no repack needed!
        im2col_u8_4b(input, ni, p.kernel_h, p.kernel_w, p.stride, p.padding, oh, ow, col_u8);

        int n_jg = (col_cols + 7) / 8;

        #pragma omp parallel for schedule(dynamic)
        for (int oc_base = 0; oc_base < p.out_channels; oc_base += OC_TILE) {
            int oc_end = std::min(oc_base + OC_TILE, (int)p.out_channels);
            int n_oc = oc_end - oc_base;

            // Per-OC weight pointers and precomputed constants
            const int8_t* wt_ptrs[16];
            float dequants[16], biases[16];
            for (int t = 0; t < n_oc; t++) {
                int oc = oc_base + t;
                wt_ptrs[t] = kd.weights_i8.data() + oc * col_rows;
                dequants[t] = act_scale * kd.oc_scale[oc];
                biases[t] = kd.oc_bias[oc];
            }

            for (int jg = 0; jg < n_jg; jg++) {
                int j = jg * 8;

                // Zero accumulators
                __m256i acc[16];
                for (int t = 0; t < n_oc; t++) acc[t] = _mm256_setzero_si256();

                // ── Add/Sub GEMM: the core loop ──
                // sign_epi32(act, w): +act if w>0, -act if w<0, 0 if w==0
                for (int k = 0; k < col_rows; k++) {
                    __m128i raw;
                    if (j + 8 <= col_cols) {
                        raw = _mm_loadl_epi64((__m128i*)(col_u8 + k * col_cols + j));
                    } else {
                        alignas(16) uint8_t tmp[8] = {};
                        for (int jj = j; jj < col_cols; jj++) tmp[jj - j] = col_u8[k * col_cols + jj];
                        raw = _mm_loadl_epi64((__m128i*)tmp);
                    }
                    __m256i act = _mm256_cvtepu8_epi32(raw);

                    for (int t = 0; t < n_oc; t++) {
                        __m256i w = _mm256_set1_epi32((int)wt_ptrs[t][k]);
                        acc[t] = _mm256_add_epi32(acc[t], _mm256_sign_epi32(act, w));
                    }
                }

                // ── Epilogue ──
                if (epilogue_mode == 0) {
                    // Mode 0: output FP32
                    for (int t = 0; t < n_oc; t++) {
                        float* dst = out_fp32->ptr(ni, oc_base + t);
                        __m256 result = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[t]),
                            _mm256_set1_ps(dequants[t]), _mm256_set1_ps(biases[t]));
                        if (j + 8 <= col_cols)
                            _mm256_storeu_ps(dst + j, result);
                        else {
                            float tmp[8]; _mm256_storeu_ps(tmp, result);
                            for (int jj = j; jj < col_cols; jj++) dst[jj] = tmp[jj - j];
                        }
                    }
                } else {
                    // Mode 1 or 2: fused epilogue → uint8 [0, 15]
                    __m256 zero = _mm256_setzero_ps();
                    float inv_out = 1.0f / out_u8_scale;
                    __m256 inv_out_v = _mm256_set1_ps(inv_out);
                    __m256i lo_i = _mm256_setzero_si256(), hi_i = _mm256_set1_epi32(15);

                    for (int t = 0; t < n_oc; t++) {
                        int oc = oc_base + t;
                        __m256 val = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[t]),
                            _mm256_set1_ps(dequants[t]), _mm256_set1_ps(biases[t]));

                        // Mode 2: add residual
                        if (epilogue_mode == 2) {
                            if (residual_u8) {
                                __m256i ru8 = _mm256_cvtepu8_epi32(
                                    _mm_loadl_epi64((__m128i*)(residual_u8->ptr(ni, oc) + j)));
                                val = _mm256_add_ps(val, _mm256_mul_ps(
                                    _mm256_cvtepi32_ps(ru8), _mm256_set1_ps(residual_scale)));
                            } else {
                                val = _mm256_add_ps(val,
                                    _mm256_loadu_ps(residual_fp32->ptr(ni, oc) + j));
                            }
                        }

                        // ReLU + requant → uint8 [0, 15]
                        val = _mm256_max_ps(val, zero);
                        __m256i i32 = _mm256_cvtps_epi32(_mm256_mul_ps(val, inv_out_v));
                        i32 = _mm256_min_epi32(_mm256_max_epi32(i32, lo_i), hi_i);

                        uint8_t* dst = out_u8->ptr(ni, oc) + j;
                        if (j + 8 <= col_cols) {
                            __m256i p16 = _mm256_packus_epi32(i32, i32);
                            p16 = _mm256_permute4x64_epi64(p16, 0xD8);
                            _mm_storel_epi64((__m128i*)dst,
                                _mm_packus_epi16(_mm256_castsi256_si128(p16),
                                                 _mm256_castsi256_si128(p16)));
                        } else {
                            alignas(32) int32_t tmp[8];
                            _mm256_storeu_si256((__m256i*)tmp, i32);
                            for (int jj = j; jj < col_cols; jj++) dst[jj - j] = (uint8_t)tmp[jj - j];
                        }
                    }
                }
            }
        }
    }
}

// ─── Utilities ──────────────────────────────────────────────────

static void relu_inplace_4b(Tensor& t) {
    int n = t.total(), i = 0;
    __m256 zero = _mm256_setzero_ps();
    for (; i + 7 < n; i += 8)
        _mm256_storeu_ps(t.data.data() + i, _mm256_max_ps(zero, _mm256_loadu_ps(t.data.data() + i)));
    for (; i < n; i++) t.data[i] = std::max(0.0f, t.data[i]);
}

// Avgpool from uint8 → FP32
static Tensor avgpool_global_u8_4b(const TensorU8& input) {
    Tensor output(input.n, input.c, 1, 1);
    int spatial = input.h * input.w;
    float inv = input.scale / spatial;
    for (int ni = 0; ni < input.n; ni++)
        for (int ci = 0; ci < input.c; ci++) {
            const uint8_t* p = input.ptr(ni, ci);
            int32_t sum = 0;
            for (int i = 0; i < spatial; i++) sum += p[i];
            output.data[ni * input.c + ci] = sum * inv;
        }
    return output;
}

static std::vector<float> fc_forward_4b(const Tensor& input, const FC& fc) {
    int in_f = fc.in_features, out_f = fc.out_features;
    std::vector<float> output(out_f);
    for (int o = 0; o < out_f; o++) {
        const float* w = fc.weight.data() + o * in_f;
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 7 < in_f; i += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(w + i), _mm256_loadu_ps(input.data.data() + i), acc);
        __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        h = _mm_add_ps(h, _mm_shuffle_ps(h, h, _MM_SHUFFLE(2, 3, 0, 1)));
        h = _mm_add_ps(h, _mm_shuffle_ps(h, h, _MM_SHUFFLE(1, 0, 3, 2)));
        float sum = _mm_cvtss_f32(h) + fc.bias[o];
        for (; i < in_f; i++) sum += w[i] * input.data[i];
        output[o] = sum;
    }
    return output;
}

// ─── ResNet-18 4-bit Forward Pass ───────────────────────────────

struct BasicBlockIndices4b { int conv1, conv2, shortcut; };

struct LayerScales4b {
    float conv1_relu_out;
    float conv2_bn_out;
    float shortcut_out;
    float block_relu_out;
};

class Engine4bit {
public:
    Model model;
    std::vector<BasicBlockIndices4b> blocks;
    std::vector<AddSubKernelData> as_kernels;

    float initial_relu_scale;
    std::vector<LayerScales4b> block_scales;

    // Preallocated workspace — only col_u8, no col_packed needed!
    std::vector<uint8_t> ws_col_u8;

    void init(const Model& m, const char* scales_path) {
        model = m;
        as_kernels.resize(model.convs.size());

        for (size_t i = 0; i < model.convs.size(); i++) {
            if (model.convs[i].is_ternary)
                prepare_addsub_kernel(model.convs[i], as_kernels[i]);
            // Shortcuts stay FP32 — no kernel prep needed
        }

        int idx = 1;
        bool has_sc[] = {false, true, true, true};
        for (int l = 0; l < 4; l++)
            for (int b = 0; b < 2; b++) {
                BasicBlockIndices4b bi;
                bi.conv1 = idx++; bi.conv2 = idx++;
                bi.shortcut = (b == 0 && has_sc[l]) ? idx++ : -1;
                blocks.push_back(bi);
            }

        load_scales(scales_path);

        // Preallocate workspace: only col_u8 (no col_packed!)
        size_t max_col_u8 = 0;
        for (size_t i = 1; i < model.convs.size(); i++) {
            if (!model.convs[i].is_ternary) continue;
            auto& kd = as_kernels[i];
            int max_spatial = 1024;  // 32×32 for CIFAR-10
            size_t cu = (size_t)kd.col_rows * max_spatial;
            if (cu > max_col_u8) max_col_u8 = cu;
        }
        ws_col_u8.resize(max_col_u8);
        printf("Workspace: col_u8=%zuKB (no col_packed — add/sub needs no repack)\n",
               max_col_u8 / 1024);

        printf("Engine4bit: %zu convs, %zu blocks, sign+add GEMM (no dpbusd)\n",
               model.convs.size(), blocks.size());
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

        block_scales.resize(blocks.size());
        const char* layer_names[] = {"layer1", "layer2", "layer3", "layer4"};
        int bi = 0;
        for (int l = 0; l < 4; l++) {
            for (int b = 0; b < 2; b++) {
                char buf[64];
                snprintf(buf, sizeof(buf), "%s.%d.conv1_relu_out", layer_names[l], b);
                block_scales[bi].conv1_relu_out = get_scale(buf);
                snprintf(buf, sizeof(buf), "%s.%d.conv2_bn_out", layer_names[l], b);
                block_scales[bi].conv2_bn_out = get_scale(buf);
                if (blocks[bi].shortcut >= 0) {
                    snprintf(buf, sizeof(buf), "%s.%d.shortcut_out", layer_names[l], b);
                    block_scales[bi].shortcut_out = get_scale(buf);
                }
                snprintf(buf, sizeof(buf), "%s.%d.block_relu_out", layer_names[l], b);
                block_scales[bi].block_relu_out = get_scale(buf);
                bi++;
            }
        }
    }

    std::vector<float> forward(const Tensor& input) {
        // First conv (FP32) + BN + ReLU → quantize to uint8 [0, 15]
        Tensor x_fp = conv2d_fp32_4b(input, model.convs[0]);
        relu_inplace_4b(x_fp);
        TensorU8 x = quantize_to_u8_4bit(x_fp, initial_relu_scale);

        // ResNet blocks — 4-bit resident
        for (size_t bi = 0; bi < blocks.size(); bi++) {
            auto& blk = blocks[bi];
            auto& sc = block_scales[bi];

            // Conv1 (ternary, add/sub) → fused ReLU + requant → uint8 [0,15]
            TensorU8 out1_u8;
            conv2d_ternary_addsub(x, model.convs[blk.conv1], as_kernels[blk.conv1],
                                  ws_col_u8.data(),
                                  1, nullptr, &out1_u8, sc.conv1_relu_out);

            // Conv2 + residual + ReLU + requant → uint8 [0,15]
            TensorU8 block_out;
            if (blk.shortcut >= 0) {
                // Shortcut: FP32 conv (not ternary, no add/sub)
                Tensor shortcut = conv2d_fp32_from_u8_4b(x, model.convs[blk.shortcut]);
                conv2d_ternary_addsub(out1_u8, model.convs[blk.conv2], as_kernels[blk.conv2],
                                      ws_col_u8.data(),
                                      2, nullptr, &block_out, sc.block_relu_out,
                                      nullptr, &shortcut, 0.0f);
            } else {
                // Identity shortcut: pass x (uint8) as residual
                conv2d_ternary_addsub(out1_u8, model.convs[blk.conv2], as_kernels[blk.conv2],
                                      ws_col_u8.data(),
                                      2, nullptr, &block_out, sc.block_relu_out,
                                      &x, nullptr, x.scale);
            }
            x = block_out;
        }

        // Avgpool (from uint8) + FC
        Tensor pooled = avgpool_global_u8_4b(x);
        return fc_forward_4b(pooled, model.fc);
    }
};
