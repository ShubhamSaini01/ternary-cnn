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
// Nibble-Packed dpbusd Ternary ResNet-18 Inference Engine
//
// Activations stored as 4-bit nibbles (TensorU4) between layers.
// im2col writes directly to spatial-packed layout (no separate repack).
// dpbusd GEMM — same proven kernel as engine.h.
// Epilogue fuses ReLU + requant + nibble pack.
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

// ─── TensorU4: nibble-packed activations ────────────────────
// Two 4-bit values per byte: high nibble = even index, low = odd.
// For CIFAR-10 ResNet: h*w is always even, so channels are byte-aligned.

struct TensorU4 {
    std::vector<uint8_t> data;
    int n, c, h, w;
    float scale;  // real_value = nibble_val * scale

    TensorU4() : n(0), c(0), h(0), w(0), scale(1.0f) {}
    TensorU4(int n, int c, int h, int w, float s)
        : data((n * c * h * w + 1) / 2, 0), n(n), c(c), h(h), w(w), scale(s) {}

    int total() const { return n * c * h * w; }
    int packed_bytes() const { return (int)data.size(); }

    uint8_t get(int abs_idx) const {
        return (abs_idx & 1) ? (data[abs_idx >> 1] & 0x0F) : (data[abs_idx >> 1] >> 4);
    }
    void set(int abs_idx, uint8_t val) {
        int b = abs_idx >> 1;
        if (abs_idx & 1) data[b] = (data[b] & 0xF0) | (val & 0x0F);
        else             data[b] = (data[b] & 0x0F) | ((val & 0x0F) << 4);
    }
};

// ─── TensorU8: workspace for unpacked activations ───────────
// Used as temporary during im2col — not stored between layers.

struct TensorU8 {
    std::vector<uint8_t> data;
    int n, c, h, w;
    float scale;
    TensorU8() : n(0), c(0), h(0), w(0), scale(1.0f) {}
    TensorU8(int n, int c, int h, int w, float s) : data(n * c * h * w, 0), n(n), c(c), h(h), w(w), scale(s) {}
    uint8_t* ptr(int ni, int ci) { return data.data() + ((ni * c + ci) * h) * w; }
    const uint8_t* ptr(int ni, int ci) const { return data.data() + ((ni * c + ci) * h) * w; }
    int total() const { return n * c * h * w; }
};

// ─── SIMD nibble unpack: TensorU4 image → TensorU8 workspace ─

static void unpack_u4_to_u8(const TensorU4& src, int ni, TensorU8& dst) {
    dst.c = src.c; dst.h = src.h; dst.w = src.w; dst.n = 1;
    dst.scale = src.scale;
    int elems = src.c * src.h * src.w;
    int base = ni * elems;
    const uint8_t* sp = src.data.data() + base / 2;
    uint8_t* dp = dst.data.data();
    int packed_bytes = elems / 2;

    __m128i mask = _mm_set1_epi8(0x0F);
    int i = 0;
    for (; i + 15 < packed_bytes; i += 16) {
        __m128i p = _mm_loadu_si128((__m128i*)(sp + i));
        __m128i hi = _mm_and_si128(_mm_srli_epi16(p, 4), mask);
        __m128i lo = _mm_and_si128(p, mask);
        _mm_storeu_si128((__m128i*)(dp + i * 2),      _mm_unpacklo_epi8(hi, lo));
        _mm_storeu_si128((__m128i*)(dp + i * 2 + 16), _mm_unpackhi_epi8(hi, lo));
    }
    for (; i < packed_bytes; i++) {
        dp[i * 2]     = (sp[i] >> 4) & 0x0F;
        dp[i * 2 + 1] = sp[i] & 0x0F;
    }
    if (elems & 1) dp[elems - 1] = (sp[packed_bytes] >> 4) & 0x0F;
}

// ─── Quantize FP32 → TensorU4 ──────────────────────────────

static TensorU4 quantize_to_u4(const Tensor& t, float out_scale) {
    TensorU4 out(t.n, t.c, t.h, t.w, out_scale);
    float inv = 1.0f / out_scale;
    int tot = t.total();
    // Process pairs for nibble packing
    for (int i = 0; i < tot - 1; i += 2) {
        int hi = std::max(0, std::min(15, (int)roundf(t.data[i] * inv)));
        int lo = std::max(0, std::min(15, (int)roundf(t.data[i + 1] * inv)));
        out.data[i >> 1] = (uint8_t)((hi << 4) | lo);
    }
    if (tot & 1) {
        int v = std::max(0, std::min(15, (int)roundf(t.data[tot - 1] * inv)));
        out.data[(tot - 1) >> 1] = (uint8_t)(v << 4);
    }
    return out;
}

// ─── Direct-to-packed im2col: U8 → spatial-packed layout ────
// Writes directly in dpbusd layout [jg][kg][32 bytes].
// Eliminates the separate repack pass entirely.

// Specialized 3×3 stride-1 pad-1
static void im2col_u8_3x3s1p1_to_packed(const TensorU8& input, int ni,
                                          int out_h, int out_w,
                                          int col_rows_pad4, uint8_t* col_packed) {
    int ic = input.c, ih = input.h, iw = input.w;
    int spatial = out_h * out_w;
    int cc8 = (spatial + 7) & ~7;
    int num_k_groups = col_rows_pad4 / 4;
    int col_rows = ic * 9;

    for (int jg = 0; jg < cc8 / 8; jg++) {
        int ohs[8], ows[8];
        bool valid[8];
        for (int lane = 0; lane < 8; lane++) {
            int j = jg * 8 + lane;
            valid[lane] = (j < spatial);
            ohs[lane] = j / out_w;
            ows[lane] = j % out_w;
        }

        for (int kg = 0; kg < num_k_groups; kg++) {
            uint8_t* out = col_packed + (jg * num_k_groups + kg) * 32;

            for (int lane = 0; lane < 8; lane++) {
                if (!valid[lane]) {
                    *(uint32_t*)(out + lane * 4) = 0;
                    continue;
                }
                int oh_l = ohs[lane], ow_l = ows[lane];

                for (int ki = 0; ki < 4; ki++) {
                    int k = kg * 4 + ki;
                    if (k >= col_rows) { out[lane * 4 + ki] = 0; continue; }
                    int c_idx = k / 9, rem = k % 9, kh = rem / 3, kw = rem % 3;
                    int ih_pos = oh_l - 1 + kh;
                    int iw_pos = ow_l - 1 + kw;
                    out[lane * 4 + ki] = (ih_pos >= 0 && ih_pos < ih && iw_pos >= 0 && iw_pos < iw)
                        ? input.ptr(ni, c_idx)[ih_pos * iw + iw_pos] : 0;
                }
            }
        }
    }
}

// Generic im2col: U8 → flat col_u8 (for non-3×3s1p1: stride-2, 1×1 shortcuts)
static void im2col_u8_generic_flat(const TensorU8& input, int ni,
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

// SIMD repack: col_u8 [k][j] → spatial-packed [jg][kg][32] (for generic case only)
static void repack_to_spatial_packed(const uint8_t* col_u8, int col_rows, int col_cols,
                                      int col_rows_pad4, uint8_t* col_packed) {
    int num_k_groups = col_rows_pad4 / 4;
    int cc8 = (col_cols + 7) & ~7;

    // Zero-pad extra k rows
    // (col_u8 buffer is assumed large enough; caller pads)

    for (int kg = 0; kg < num_k_groups; kg++) {
        const uint8_t* r0 = col_u8 + (kg * 4 + 0) * col_cols;
        const uint8_t* r1 = (kg * 4 + 1 < col_rows_pad4) ? col_u8 + (kg * 4 + 1) * col_cols : nullptr;
        const uint8_t* r2 = (kg * 4 + 2 < col_rows_pad4) ? col_u8 + (kg * 4 + 2) * col_cols : nullptr;
        const uint8_t* r3 = (kg * 4 + 3 < col_rows_pad4) ? col_u8 + (kg * 4 + 3) * col_cols : nullptr;

        int jg = 0;
        for (; jg < col_cols / 8; jg++) {
            uint8_t* out = col_packed + (jg * num_k_groups + kg) * 32;
            int j = jg * 8;
            __m128i v0 = _mm_loadl_epi64((__m128i*)(r0 + j));
            __m128i v1 = r1 ? _mm_loadl_epi64((__m128i*)(r1 + j)) : _mm_setzero_si128();
            __m128i v2 = r2 ? _mm_loadl_epi64((__m128i*)(r2 + j)) : _mm_setzero_si128();
            __m128i v3 = r3 ? _mm_loadl_epi64((__m128i*)(r3 + j)) : _mm_setzero_si128();

            __m128i lo01 = _mm_unpacklo_epi8(v0, v1);
            __m128i lo23 = _mm_unpacklo_epi8(v2, v3);
            _mm_storeu_si128((__m128i*)(out),      _mm_unpacklo_epi16(lo01, lo23));
            _mm_storeu_si128((__m128i*)(out + 16), _mm_unpackhi_epi16(lo01, lo23));
        }
        // Tail
        for (int jg2 = jg; jg2 < cc8 / 8; jg2++) {
            uint8_t* out = col_packed + (jg2 * num_k_groups + kg) * 32;
            for (int lane = 0; lane < 8; lane++) {
                int j = jg2 * 8 + lane;
                if (j < col_cols) {
                    out[lane * 4 + 0] = r0[j];
                    out[lane * 4 + 1] = r1 ? r1[j] : 0;
                    out[lane * 4 + 2] = r2 ? r2[j] : 0;
                    out[lane * 4 + 3] = r3 ? r3[j] : 0;
                } else {
                    *(uint32_t*)(out + lane * 4) = 0;
                }
            }
        }
    }
}

// ─── FP32 im2col + conv (first conv only) ──────────────────

static void im2col_fp32(const Tensor& input, int ni,
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

static Tensor conv2d_fp32(const Tensor& input, const FusedConv& layer) {
    auto& p = layer.p;
    int oh = (input.h + 2 * p.padding - p.kernel_h) / p.stride + 1;
    int ow = (input.w + 2 * p.padding - p.kernel_w) / p.stride + 1;
    Tensor output(input.n, p.out_channels, oh, ow);
    int col_rows = p.in_channels * p.kernel_h * p.kernel_w;
    int col_cols = oh * ow;
    std::vector<float> col(col_rows * col_cols);

    for (int ni = 0; ni < input.n; ni++) {
        im2col_fp32(input, ni, p.kernel_h, p.kernel_w, p.stride, p.padding, oh, ow, col.data());
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

// ─── dpbusd Kernel Data (same as engine.h) ──────────────────

struct I8KernelData {
    int col_rows, col_rows_pad4, num_k_groups;
    std::vector<int32_t> weights_packed;  // [OC][num_groups] packed int32
    std::vector<float> oc_scale;
    std::vector<float> oc_bias;
};

static int32_t pack4_i8(int8_t w0, int8_t w1, int8_t w2, int8_t w3) {
    return ((int32_t)(uint8_t)w0) | ((int32_t)(uint8_t)w1 << 8) |
           ((int32_t)(uint8_t)w2 << 16) | ((int32_t)(uint8_t)w3 << 24);
}

static void prepare_ternary_kernel(const FusedConv& layer, I8KernelData& kd) {
    int noc = layer.p.out_channels;
    int ic = layer.p.in_channels, kH = layer.p.kernel_h, kW = layer.p.kernel_w;
    kd.col_rows = ic * kH * kW;
    kd.col_rows_pad4 = (kd.col_rows + 3) & ~3;
    int ng = kd.col_rows_pad4 / 4;
    kd.num_k_groups = ng;

    std::vector<int8_t> w_padded(noc * kd.col_rows_pad4, 0);
    for (int o = 0; o < noc; o++)
        memcpy(w_padded.data() + o * kd.col_rows_pad4,
               layer.weights_i8.data() + o * kd.col_rows, kd.col_rows);

    kd.weights_packed.resize(noc * ng);
    for (int o = 0; o < noc; o++) {
        const int8_t* w = w_padded.data() + o * kd.col_rows_pad4;
        for (int g = 0; g < ng; g++)
            kd.weights_packed[o * ng + g] = pack4_i8(w[g * 4], w[g * 4 + 1], w[g * 4 + 2], w[g * 4 + 3]);
    }

    kd.oc_scale.resize(noc);
    kd.oc_bias.resize(noc);
    for (int o = 0; o < noc; o++) {
        kd.oc_scale[o] = layer.alpha[o] * layer.fused_scale[o];
        kd.oc_bias[o] = layer.fused_bias[o];
    }
}

static void prepare_shortcut_kernel(const FusedConv& layer, I8KernelData& kd) {
    int noc = layer.p.out_channels;
    kd.col_rows = layer.p.in_channels * layer.p.kernel_h * layer.p.kernel_w;
    kd.col_rows_pad4 = (kd.col_rows + 3) & ~3;
    int ng = kd.col_rows_pad4 / 4;
    kd.num_k_groups = ng;

    std::vector<int8_t> w_i8(noc * kd.col_rows_pad4, 0);
    std::vector<float> w_scale(noc);

    for (int o = 0; o < noc; o++) {
        const float* w = layer.weights_fp32.data() + o * kd.col_rows;
        float maxabs = 0;
        for (int k = 0; k < kd.col_rows; k++) {
            float v = fabsf(w[k]);
            if (v > maxabs) maxabs = v;
        }
        w_scale[o] = (maxabs > 1e-10f) ? maxabs / 127.0f : 1.0f;
        float inv = 1.0f / w_scale[o];
        int8_t* dst = w_i8.data() + o * kd.col_rows_pad4;
        for (int k = 0; k < kd.col_rows; k++)
            dst[k] = (int8_t)std::max(-128, std::min(127, (int)roundf(w[k] * inv)));
    }

    kd.weights_packed.resize(noc * ng);
    for (int o = 0; o < noc; o++) {
        const int8_t* w = w_i8.data() + o * kd.col_rows_pad4;
        for (int g = 0; g < ng; g++)
            kd.weights_packed[o * ng + g] = pack4_i8(w[g * 4], w[g * 4 + 1], w[g * 4 + 2], w[g * 4 + 3]);
    }

    kd.oc_scale.resize(noc);
    kd.oc_bias.resize(noc);
    for (int o = 0; o < noc; o++) {
        kd.oc_scale[o] = w_scale[o] * layer.fused_scale[o];
        kd.oc_bias[o] = layer.fused_bias[o];
    }
}

// ─── dpbusd Conv from U4: unpack → im2col → GEMM → pack ────
//
// For 3×3s1p1: unpack U4→U8, then direct-to-packed im2col (no repack)
// For generic: unpack U4→U8, im2col→col_u8, repack→col_packed
// dpbusd GEMM is identical in both cases.
//
// Epilogue modes:
//   0: output FP32 (out_fp32) — for shortcut
//   1: fused ReLU + requant → U4 (out_u4) — for conv1
//   2: fused residual + ReLU + requant → U4 (out_u4) — for conv2

static void conv2d_dpbusd_from_u4(const TensorU4& input, const FusedConv& layer,
                                   const I8KernelData& kd,
                                   TensorU8& ws_unpacked,
                                   uint8_t* col_u8, uint8_t* col_packed,
                                   int epilogue_mode,
                                   Tensor* out_fp32,
                                   TensorU4* out_u4, float out_u4_scale,
                                   const TensorU4* residual_u4 = nullptr,
                                   const Tensor* residual_fp32 = nullptr,
                                   float residual_scale = 0.0f) {
    auto& p = layer.p;
    int oh = (input.h + 2 * p.padding - p.kernel_h) / p.stride + 1;
    int ow = (input.w + 2 * p.padding - p.kernel_w) / p.stride + 1;

    if (epilogue_mode == 0) *out_fp32 = Tensor(input.n, p.out_channels, oh, ow);
    else *out_u4 = TensorU4(input.n, p.out_channels, oh, ow, out_u4_scale);

    int col_rows = kd.col_rows;
    int col_rows_pad4 = kd.col_rows_pad4;
    int num_k_groups = kd.num_k_groups;
    int col_cols = oh * ow;
    int cc8 = (col_cols + 7) & ~7;
    float act_scale = input.scale;

    bool is_3x3s1p1 = (p.kernel_h == 3 && p.kernel_w == 3 && p.stride == 1 && p.padding == 1);

    for (int ni = 0; ni < input.n; ni++) {
        // Step 1: Unpack U4 → U8 workspace
        unpack_u4_to_u8(input, ni, ws_unpacked);

        // Step 2: im2col → spatial-packed
        if (is_3x3s1p1) {
            // Fused: direct to spatial-packed layout (no col_u8, no repack)
            im2col_u8_3x3s1p1_to_packed(ws_unpacked, 0, oh, ow, col_rows_pad4, col_packed);
        } else {
            // Generic: im2col → col_u8 → repack
            im2col_u8_generic_flat(ws_unpacked, 0, p.kernel_h, p.kernel_w,
                                    p.stride, p.padding, oh, ow, col_u8);
            // Zero-pad extra k rows
            for (int k = col_rows; k < col_rows_pad4; k++)
                memset(col_u8 + k * col_cols, 0, col_cols);
            repack_to_spatial_packed(col_u8, col_rows, col_cols, col_rows_pad4, col_packed);
        }

        // Step 3: dpbusd GEMM (identical to engine.h)
        const int OC_TILE = 16;
        int n_jg = cc8 / 8;

        #pragma omp parallel for schedule(dynamic)
        for (int oc_base = 0; oc_base < p.out_channels; oc_base += OC_TILE) {
            int oc_end = std::min(oc_base + OC_TILE, (int)p.out_channels);
            int n_oc = oc_end - oc_base;

            const int32_t* wpk_ptrs[16];
            float dequants[16], biases[16];
            for (int t = 0; t < n_oc; t++) {
                int oc = oc_base + t;
                wpk_ptrs[t] = kd.weights_packed.data() + oc * num_k_groups;
                dequants[t] = act_scale * kd.oc_scale[oc];
                biases[t] = kd.oc_bias[oc];
            }

            for (int jg = 0; jg < n_jg; jg++) {
                __m256i acc[16];
                for (int t = 0; t < n_oc; t++) acc[t] = _mm256_setzero_si256();

                for (int kg = 0; kg < num_k_groups; kg++) {
                    __m256i a = _mm256_loadu_si256((__m256i*)(col_packed + (jg * num_k_groups + kg) * 32));
                    for (int t = 0; t < n_oc; t++)
                        acc[t] = _mm256_dpbusd_epi32(acc[t], a, _mm256_set1_epi32(wpk_ptrs[t][kg]));
                }

                int j = jg * 8;

                // Step 4: Epilogue
                if (epilogue_mode == 0) {
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
                    // Fused ReLU + requant → U4 (nibble pack)
                    __m256 zero = _mm256_setzero_ps();
                    float inv_out = 1.0f / out_u4_scale;
                    __m256 inv_out_v = _mm256_set1_ps(inv_out);
                    __m256i lo_i = _mm256_setzero_si256(), hi_i = _mm256_set1_epi32(15);

                    for (int t = 0; t < n_oc; t++) {
                        int oc = oc_base + t;
                        __m256 val = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[t]),
                            _mm256_set1_ps(dequants[t]), _mm256_set1_ps(biases[t]));

                        // Mode 2: add residual
                        if (epilogue_mode == 2) {
                            if (residual_u4) {
                                // Dequant residual U4 nibbles → FP32
                                int r_base = (ni * residual_u4->c + oc) * residual_u4->h * residual_u4->w;
                                alignas(16) uint8_t r_vals[8];
                                for (int jj = 0; jj < 8 && j + jj < col_cols; jj++)
                                    r_vals[jj] = residual_u4->get(r_base + j + jj);
                                __m256i ru4 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)r_vals));
                                val = _mm256_add_ps(val, _mm256_mul_ps(
                                    _mm256_cvtepi32_ps(ru4), _mm256_set1_ps(residual_scale)));
                            } else {
                                val = _mm256_add_ps(val,
                                    _mm256_loadu_ps(residual_fp32->ptr(ni, oc) + j));
                            }
                        }

                        // ReLU
                        val = _mm256_max_ps(val, zero);
                        // Requant to [0, 15]
                        __m256i i32 = _mm256_cvtps_epi32(_mm256_mul_ps(val, inv_out_v));
                        i32 = _mm256_min_epi32(_mm256_max_epi32(i32, lo_i), hi_i);

                        // Pack to nibbles and store into TensorU4
                        int base_idx = (ni * out_u4->c + oc) * (out_u4->h * out_u4->w) + j;
                        if (j + 8 <= col_cols) {
                            // Fast path: pack i32 → u8 → nibble pairs
                            __m256i p16 = _mm256_packus_epi32(i32, i32);
                            p16 = _mm256_permute4x64_epi64(p16, 0xD8);
                            __m128i p8 = _mm_packus_epi16(
                                _mm256_castsi256_si128(p16), _mm256_castsi256_si128(p16));
                            alignas(16) uint8_t vals[8];
                            _mm_storel_epi64((__m128i*)vals, p8);

                            uint8_t* dst = out_u4->data.data() + base_idx / 2;
                            dst[0] = (vals[0] << 4) | vals[1];
                            dst[1] = (vals[2] << 4) | vals[3];
                            dst[2] = (vals[4] << 4) | vals[5];
                            dst[3] = (vals[6] << 4) | vals[7];
                        } else {
                            alignas(32) int32_t tmp[8];
                            _mm256_storeu_si256((__m256i*)tmp, i32);
                            for (int jj = j; jj < col_cols; jj++)
                                out_u4->set(base_idx + jj - j, (uint8_t)tmp[jj - j]);
                        }
                    }
                }
            }
        }
    }
}

// ─── Utilities ──────────────────────────────────────────────

static void relu_inplace(Tensor& t) {
    int n = t.total(), i = 0;
    __m256 zero = _mm256_setzero_ps();
    for (; i + 7 < n; i += 8)
        _mm256_storeu_ps(t.data.data() + i, _mm256_max_ps(zero, _mm256_loadu_ps(t.data.data() + i)));
    for (; i < n; i++) t.data[i] = std::max(0.0f, t.data[i]);
}

static Tensor avgpool_global_u4(const TensorU4& input) {
    Tensor output(input.n, input.c, 1, 1);
    int spatial = input.h * input.w;
    float inv = input.scale / spatial;
    for (int ni = 0; ni < input.n; ni++)
        for (int ci = 0; ci < input.c; ci++) {
            int base = (ni * input.c + ci) * spatial;
            int32_t sum = 0;
            // Unpack nibbles and sum
            int packed_bytes = spatial / 2;
            const uint8_t* src = input.data.data() + base / 2;
            for (int i = 0; i < packed_bytes; i++) {
                sum += (src[i] >> 4);       // high nibble
                sum += (src[i] & 0x0F);     // low nibble
            }
            if (spatial & 1) sum += input.get(base + spatial - 1);
            output.data[ni * input.c + ci] = sum * inv;
        }
    return output;
}

static std::vector<float> fc_forward(const Tensor& input, const FC& fc) {
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

// ─── Engine ─────────────────────────────────────────────────

struct BasicBlockIndices { int conv1, conv2, shortcut; };

struct LayerScales {
    float conv1_relu_out;
    float conv2_bn_out;
    float shortcut_out;
    float block_relu_out;
};

class Engine4bitDPBUSD {
public:
    Model model;
    std::vector<BasicBlockIndices> blocks;
    std::vector<I8KernelData> i8_kernels;

    float initial_relu_scale;
    std::vector<LayerScales> block_scales;

    // Workspaces
    TensorU8 ws_unpacked;     // for U4→U8 unpack
    std::vector<uint8_t> ws_col_u8;      // for generic im2col
    std::vector<uint8_t> ws_col_packed;  // spatial-packed for dpbusd

    void init(const Model& m, const char* scales_path) {
        model = m;
        i8_kernels.resize(model.convs.size());
        for (size_t i = 0; i < model.convs.size(); i++) {
            if (model.convs[i].is_ternary)
                prepare_ternary_kernel(model.convs[i], i8_kernels[i]);
            else if (i > 0)
                prepare_shortcut_kernel(model.convs[i], i8_kernels[i]);
        }

        int idx = 1;
        bool has_sc[] = {false, true, true, true};
        for (int l = 0; l < 4; l++)
            for (int b = 0; b < 2; b++) {
                BasicBlockIndices bi;
                bi.conv1 = idx++; bi.conv2 = idx++;
                bi.shortcut = (b == 0 && has_sc[l]) ? idx++ : -1;
                blocks.push_back(bi);
            }

        load_scales(scales_path);

        // Preallocate workspaces
        size_t max_unpacked = 0, max_col_u8 = 0, max_col_packed = 0;
        for (size_t i = 1; i < model.convs.size(); i++) {
            auto& kd = i8_kernels[i];
            if (kd.col_rows == 0) continue;
            auto& p = model.convs[i].p;
            int max_spatial = 1024;
            int cc8 = (max_spatial + 7) & ~7;

            size_t cp = (size_t)(cc8 / 8) * kd.num_k_groups * 32;
            if (cp > max_col_packed) max_col_packed = cp;

            bool is_3x3s1p1 = (p.kernel_h == 3 && p.kernel_w == 3 && p.stride == 1 && p.padding == 1);
            if (!is_3x3s1p1) {
                size_t cu = (size_t)kd.col_rows_pad4 * max_spatial;
                if (cu > max_col_u8) max_col_u8 = cu;
            }

            // Input tensor size for unpack workspace
            size_t inp_size = (size_t)p.in_channels * 32 * 32;  // max input spatial
            if (inp_size > max_unpacked) max_unpacked = inp_size;
        }

        ws_unpacked.data.resize(max_unpacked);
        ws_col_u8.resize(max_col_u8);
        ws_col_packed.resize(max_col_packed);

        printf("Workspace: unpacked=%zuKB col_u8=%zuKB col_packed=%zuKB\n",
               max_unpacked / 1024, max_col_u8 / 1024, max_col_packed / 1024);
        printf("Engine4bitDPBUSD: %zu convs, %zu blocks, U4 nibble-packed + dpbusd\n",
               model.convs.size(), blocks.size());
        printf("  3x3s1p1 layers: direct-to-packed im2col (no repack)\n");
        printf("  Model memory: %.2f KB (nibble-packed activations)\n",
               max_unpacked / 2.0 / 1024.0);
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
        // First conv (FP32) + BN + ReLU → U4
        Tensor x_fp = conv2d_fp32(input, model.convs[0]);
        relu_inplace(x_fp);
        TensorU4 x = quantize_to_u4(x_fp, initial_relu_scale);

        // ResNet blocks — U4 resident
        for (size_t bi = 0; bi < blocks.size(); bi++) {
            auto& blk = blocks[bi];
            auto& sc = block_scales[bi];

            // Conv1 → fused ReLU + requant → U4
            TensorU4 out1_u4;
            conv2d_dpbusd_from_u4(x, model.convs[blk.conv1], i8_kernels[blk.conv1],
                                   ws_unpacked, ws_col_u8.data(), ws_col_packed.data(),
                                   1, nullptr, &out1_u4, sc.conv1_relu_out);

            // Conv2 + residual → U4
            TensorU4 block_out;
            if (blk.shortcut >= 0) {
                // Shortcut via dpbusd (mode 0 → FP32)
                Tensor shortcut;
                conv2d_dpbusd_from_u4(x, model.convs[blk.shortcut], i8_kernels[blk.shortcut],
                                       ws_unpacked, ws_col_u8.data(), ws_col_packed.data(),
                                       0, &shortcut, nullptr, 0.0f);
                conv2d_dpbusd_from_u4(out1_u4, model.convs[blk.conv2], i8_kernels[blk.conv2],
                                       ws_unpacked, ws_col_u8.data(), ws_col_packed.data(),
                                       2, nullptr, &block_out, sc.block_relu_out,
                                       nullptr, &shortcut, 0.0f);
            } else {
                conv2d_dpbusd_from_u4(out1_u4, model.convs[blk.conv2], i8_kernels[blk.conv2],
                                       ws_unpacked, ws_col_u8.data(), ws_col_packed.data(),
                                       2, nullptr, &block_out, sc.block_relu_out,
                                       &x, nullptr, x.scale);
            }
            x = block_out;
        }

        // Avgpool from U4 → FP32 → FC
        Tensor pooled = avgpool_global_u4(x);
        return fc_forward(pooled, model.fc);
    }
};
