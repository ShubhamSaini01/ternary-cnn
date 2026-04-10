#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <immintrin.h>
#include <omp.h>
#include "model_loader.h"

// ═══════════════════════════════════════════════════════════════
// INT8-Resident Ternary ResNet-18 Inference Engine
//
// Activations stay in uint8 between layers (post-ReLU = unsigned).
// Static calibrated scales — zero runtime quantization overhead.
// im2col operates on bytes (4× less data than FP32).
// dpbusd GEMM with spatial-packed layout + OC tiling.
// Only FP32 at: first conv, shortcut convs, residual add points.
// ═══════════════════════════════════════════════════════════════

// Allocator that skips zero-initialization for POD types.
// std::vector<T>(n) value-initializes (zeros) every element.
// For output buffers that the GEMM epilogue fully overwrites,
// this zeroing is pure waste (showed up as 1.4% in VTune).
template <typename T>
struct DefaultInitAlloc {
    using value_type = T;
    DefaultInitAlloc() noexcept = default;
    template <typename U> DefaultInitAlloc(const DefaultInitAlloc<U>&) noexcept {}
    T* allocate(size_t n) { return static_cast<T*>(::operator new(n * sizeof(T))); }
    void deallocate(T* p, size_t) noexcept { ::operator delete(p); }
    // Default-init (no zeroing) instead of value-init
    template <typename U> void construct(U* p) noexcept { ::new (static_cast<void*>(p)) U; }
    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) { ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...); }
};
template <typename T, typename U>
bool operator==(const DefaultInitAlloc<T>&, const DefaultInitAlloc<U>&) { return true; }
template <typename T, typename U>
bool operator!=(const DefaultInitAlloc<T>&, const DefaultInitAlloc<U>&) { return false; }

template <typename T> using FastVec = std::vector<T, DefaultInitAlloc<T>>;

struct Tensor {
    FastVec<float> data;
    int n, c, h, w;
    Tensor() : n(0), c(0), h(0), w(0) {}
    Tensor(int n, int c, int h, int w) : data(n * c * h * w), n(n), c(c), h(h), w(w) {}
    float* ptr(int ni, int ci) { return data.data() + ((ni * c + ci) * h) * w; }
    const float* ptr(int ni, int ci) const { return data.data() + ((ni * c + ci) * h) * w; }
    int total() const { return n * c * h * w; }
    // Reshape without realloc if capacity suffices (zero malloc on hot path)
    void reshape(int n_, int c_, int h_, int w_) {
        n = n_; c = c_; h = h_; w = w_;
        size_t need = (size_t)n * c * h * w;
        if (data.size() < need) data.resize(need);
    }
};

struct TensorU8 {
    FastVec<uint8_t> data;
    int n, c, h, w;
    float scale;  // real_value = (uint8_value - zero_point) * scale
    // For post-ReLU: zero_point = 0, scale = max/255
    TensorU8() : n(0), c(0), h(0), w(0), scale(1.0f) {}
    TensorU8(int n, int c, int h, int w, float s) : data(n * c * h * w), n(n), c(c), h(h), w(w), scale(s) {}
    uint8_t* ptr(int ni, int ci) { return data.data() + ((ni * c + ci) * h) * w; }
    const uint8_t* ptr(int ni, int ci) const { return data.data() + ((ni * c + ci) * h) * w; }
    int total() const { return n * c * h * w; }
    // Reshape without realloc if capacity suffices (zero malloc on hot path)
    void reshape(int n_, int c_, int h_, int w_, float s) {
        n = n_; c = c_; h = h_; w = w_; scale = s;
        size_t need = (size_t)n * c * h * w;
        if (data.size() < need) data.resize(need);
    }
};

// ─── FP32 quantize to uint8 (post-ReLU: all values ≥ 0) ────

static void quantize_to_u8(const Tensor& t, float out_scale, TensorU8& out) {
    out.reshape(t.n, t.c, t.h, t.w, out_scale);
    float inv = 1.0f / out_scale;
    int tot = t.total(), i = 0;
    __m256 sv = _mm256_set1_ps(inv);
    __m256i lo = _mm256_setzero_si256(), hi = _mm256_set1_epi32(255);
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
        out.data[i] = (uint8_t)std::max(0, std::min(255, v));
    }
}

// ─── im2col on uint8 data ───────────────────────────────────

// Generic im2col (fallback)
static void im2col_u8_generic(const TensorU8& input, int ni,
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

// Specialized 3×3 stride-1 pad-1 im2col: memcpy per row, no per-element branches
// Each im2col "row" for kernel pos (kh,kw) is a shifted copy of an input row
static void im2col_u8_3x3s1p1(const TensorU8& input, int ni,
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

// ─── Fused im2col+repack for 3×3 stride-1 pad-1 ────────────
// Skips col_u8 entirely: reads input → SSE interleave → col_packed in one pass.
// For each group of 4 reduction rows and 8 spatial positions, loads 4×8 bytes
// directly from (shifted) input rows and produces 32 bytes of spatial-packed output.
// Requires out_w divisible by 8 (true for CIFAR-10: 32, 16, 8).
static void fused_im2col_repack_3x3s1p1(const TensorU8& input, int ni,
                                          int out_h, int out_w,
                                          int col_rows_pad4, uint8_t* col_packed) {
    int ic = input.c, ih = input.h, iw = input.w;
    int col_rows = ic * 9;
    int nkg = col_rows_pad4 / 4;
    int n_jg = out_w / 8;  // spatial groups of 8 within a row (out_w must be %8==0)

    // Precompute: for each reduction index k, what is (c, kh, kw)?
    // k = c*9 + kh*3 + kw, so c = k/9, kh = (k%9)/3, kw = k%3
    // For a given output position (oh, ow):
    //   ih_val = oh + kh - 1,  iw_start = ow + kw - 1
    //   value = input[c][ih_val][iw_start] (or 0 if out of bounds)

    // Process 4 reduction rows at a time (matching repack group size)
    for (int kg = 0; kg < nkg; kg++) {
        // Decode the 4 reduction indices in this group
        int k[4];
        int c_[4], kh_[4], kw_[4];
        for (int i = 0; i < 4; i++) {
            k[i] = kg * 4 + i;
            if (k[i] < col_rows) {
                c_[i] = k[i] / 9;
                kh_[i] = (k[i] % 9) / 3;
                kw_[i] = k[i] % 3;
            } else {
                c_[i] = -1;  // padding row (beyond col_rows)
            }
        }

        // For each output row
        for (int oh = 0; oh < out_h; oh++) {
            // For each of the 4 rows, compute input row pointer (or null for padding)
            const uint8_t* rp[4];
            int iw_off[4];  // column offset: kw - 1
            for (int i = 0; i < 4; i++) {
                if (c_[i] < 0) { rp[i] = nullptr; continue; }
                int ih_val = oh + kh_[i] - 1;
                if (ih_val < 0 || ih_val >= ih) { rp[i] = nullptr; continue; }
                rp[i] = input.ptr(ni, c_[i]) + ih_val * iw;
                iw_off[i] = kw_[i] - 1;  // -1, 0, or +1
            }

            // Process 8 spatial positions at a time within this output row
            for (int owg = 0; owg < n_jg; owg++) {
                int ow = owg * 8;
                int jg_idx = oh * n_jg + owg;  // spatial group index
                uint8_t* out = col_packed + (jg_idx * nkg + kg) * 32;

                // Load 8 bytes from each of 4 rows
                __m128i v[4];
                for (int i = 0; i < 4; i++) {
                    if (!rp[i]) {
                        v[i] = _mm_setzero_si128();
                    } else {
                        int col_start = ow + iw_off[i];
                        // col_start ranges from -1 to out_w (=iw)
                        // For middle groups: col_start >= 0 && col_start+7 < iw → direct load
                        if (col_start >= 0 && col_start + 8 <= iw) {
                            v[i] = _mm_loadl_epi64((__m128i*)(rp[i] + col_start));
                        } else {
                            // Edge case: partial padding (first/last group per row)
                            alignas(16) uint8_t tmp[8] = {0,0,0,0,0,0,0,0};
                            for (int b = 0; b < 8; b++) {
                                int c_idx = col_start + b;
                                if (c_idx >= 0 && c_idx < iw) tmp[b] = rp[i][c_idx];
                            }
                            v[i] = _mm_loadl_epi64((__m128i*)tmp);
                        }
                    }
                }

                // SSE interleave (identical to repack)
                __m128i lo01 = _mm_unpacklo_epi8(v[0], v[1]);
                __m128i lo23 = _mm_unpacklo_epi8(v[2], v[3]);
                _mm_storeu_si128((__m128i*)(out), _mm_unpacklo_epi16(lo01, lo23));
                _mm_storeu_si128((__m128i*)(out + 16), _mm_unpackhi_epi16(lo01, lo23));
            }
        }
    }
}

// Dispatch: use specialized version when applicable
static void im2col_u8(const TensorU8& input, int ni,
                       int kH, int kW, int stride, int padding,
                       int out_h, int out_w, uint8_t* col) {
    if (kH == 3 && kW == 3 && stride == 1 && padding == 1) {
        im2col_u8_3x3s1p1(input, ni, out_h, out_w, col);
    } else {
        im2col_u8_generic(input, ni, kH, kW, stride, padding, out_h, out_w, col);
    }
}

// Direct-to-VNNI im2col for 3×3 stride 1 pad 1:
// Writes directly in spatial-packed layout [jg][kg][32 bytes]
// Eliminates the separate repack pass entirely
static void im2col_u8_3x3s1p1_packed(const TensorU8& input, int ni,
                                       int out_h, int out_w,
                                       int col_rows_pad4, uint8_t* col_packed) {
    int ic = input.c, ih = input.h, iw = input.w;
    int spatial = out_h * out_w;
    int cc8 = (spatial + 7) & ~7;
    int num_k_groups = col_rows_pad4 / 4;
    int col_rows = ic * 9;

    // For each (jg, kg) block: write 32 bytes = 8 spatial × 4 reduction
    // Reduction index k maps to: ic = k/9, kh = (k%9)/3, kw = k%3
    // For spatial j: oh = j/out_w, ow = j%out_w
    // Input access: input[ic][oh - 1 + kh][ow - 1 + kw]

    for (int jg = 0; jg < cc8/8; jg++) {
        // Precompute oh, ow for these 8 spatial positions
        int ohs[8], ows[8];
        bool valid[8];
        for (int lane = 0; lane < 8; lane++) {
            int j = jg*8 + lane;
            valid[lane] = (j < spatial);
            ohs[lane] = j / out_w;
            ows[lane] = j % out_w;
        }

        for (int kg = 0; kg < num_k_groups; kg++) {
            uint8_t* out = col_packed + (jg * num_k_groups + kg) * 32;

            for (int lane = 0; lane < 8; lane++) {
                if (!valid[lane]) {
                    *(uint32_t*)(out + lane*4) = 0;
                    continue;
                }
                int oh_l = ohs[lane], ow_l = ows[lane];

                for (int ki = 0; ki < 4; ki++) {
                    int k = kg*4 + ki;
                    if (k >= col_rows) { out[lane*4+ki] = 0; continue; }
                    int c = k / 9, rem = k % 9, kh = rem / 3, kw = rem % 3;
                    int ih_pos = oh_l - 1 + kh;
                    int iw_pos = ow_l - 1 + kw;
                    out[lane*4+ki] = (ih_pos >= 0 && ih_pos < ih && iw_pos >= 0 && iw_pos < iw)
                        ? input.ptr(ni, c)[ih_pos * iw + iw_pos] : 0;
                }
            }
        }
    }
}

// ─── FP32 im2col (for first conv + shortcuts) ──────────────

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

// ─── FP32 Conv (first conv + shortcuts) ─────────────────────

static void conv2d_fp32(const Tensor& input, const FusedConv& layer, Tensor& output) {
    auto& p = layer.p;
    int oh = (input.h + 2 * p.padding - p.kernel_h) / p.stride + 1;
    int ow = (input.w + 2 * p.padding - p.kernel_w) / p.stride + 1;
    output.reshape(input.n, p.out_channels, oh, ow);
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
}

// FP32 conv from uint8 input (for shortcuts)
static Tensor conv2d_fp32_from_u8(const TensorU8& input, const FusedConv& layer) {
    auto& p = layer.p;
    int oh = (input.h + 2 * p.padding - p.kernel_h) / p.stride + 1;
    int ow = (input.w + 2 * p.padding - p.kernel_w) / p.stride + 1;
    Tensor output(input.n, p.out_channels, oh, ow);
    int col_rows = p.in_channels * p.kernel_h * p.kernel_w;
    int col_cols = oh * ow;

    std::vector<float> col(col_rows * col_cols);
    std::vector<uint8_t> col_u8(col_rows * col_cols);
    for (int ni = 0; ni < input.n; ni++) {
        im2col_u8(input, ni, p.kernel_h, p.kernel_w, p.stride, p.padding, oh, ow, col_u8.data());
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

// ─── Ternary Conv: INT8-resident, spatial-packed dpbusd ─────

struct I8KernelData {
    int col_rows, col_rows_pad4, num_k_groups;
    std::vector<int32_t> weights_packed;  // [OC][num_groups] as packed int32

    // Precomputed per-OC constants (avoids recomputing per inference)
    std::vector<float> oc_scale;  // alpha[oc] * fused_bn_scale[oc] (for ternary), or weight_scale[oc] * fused_bn_scale (for shortcut)
    std::vector<float> oc_bias;   // fused_bn_bias[oc]
};

// Helper: pack 4 int8 weights into one int32 for dpbusd broadcast
static int32_t pack4_i8(int8_t w0, int8_t w1, int8_t w2, int8_t w3) {
    return ((int32_t)(uint8_t)w0) | ((int32_t)(uint8_t)w1<<8) |
           ((int32_t)(uint8_t)w2<<16) | ((int32_t)(uint8_t)w3<<24);
}

static void prepare_ternary_kernel(const FusedConv& layer, I8KernelData& kd) {
    int noc = layer.p.out_channels;
    int ic = layer.p.in_channels, kH = layer.p.kernel_h, kW = layer.p.kernel_w;
    kd.col_rows = ic * kH * kW;
    kd.col_rows_pad4 = (kd.col_rows + 3) & ~3;
    int ng = kd.col_rows_pad4 / 4;
    kd.num_k_groups = ng;

    // Pad weights to multiple of 4
    std::vector<int8_t> w_padded(noc * kd.col_rows_pad4, 0);
    for (int o = 0; o < noc; o++) {
        memcpy(w_padded.data() + o * kd.col_rows_pad4,
               layer.weights_i8.data() + o * kd.col_rows, kd.col_rows);
    }

    kd.weights_packed.resize(noc * ng);
    for (int o = 0; o < noc; o++) {
        const int8_t* w = w_padded.data() + o * kd.col_rows_pad4;
        for (int g = 0; g < ng; g++)
            kd.weights_packed[o*ng+g] = pack4_i8(w[g*4], w[g*4+1], w[g*4+2], w[g*4+3]);
    }

    // Precompute per-OC constants
    kd.oc_scale.resize(noc);
    kd.oc_bias.resize(noc);
    for (int o = 0; o < noc; o++) {
        kd.oc_scale[o] = layer.alpha[o] * layer.fused_scale[o];
        kd.oc_bias[o] = layer.fused_bias[o];
    }
}

// Prepare shortcut conv: quantize FP32 weights to INT8, build same dpbusd packed format
static void prepare_shortcut_kernel(const FusedConv& layer, I8KernelData& kd) {
    int noc = layer.p.out_channels;
    kd.col_rows = layer.p.in_channels * layer.p.kernel_h * layer.p.kernel_w;
    kd.col_rows_pad4 = (kd.col_rows + 3) & ~3;
    int ng = kd.col_rows_pad4 / 4;
    kd.num_k_groups = ng;

    // Quantize FP32 weights to INT8: per-OC symmetric quantization
    // weight_i8[o][k] = round(weight_fp32[o][k] / w_scale[o])
    // w_scale[o] = max(|w[o][:]|) / 127
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
        for (int k = 0; k < kd.col_rows; k++) {
            int v = (int)roundf(w[k] * inv);
            dst[k] = (int8_t)std::max(-128, std::min(127, v));
        }
    }

    kd.weights_packed.resize(noc * ng);
    for (int o = 0; o < noc; o++) {
        const int8_t* w = w_i8.data() + o * kd.col_rows_pad4;
        for (int g = 0; g < ng; g++)
            kd.weights_packed[o*ng+g] = pack4_i8(w[g*4], w[g*4+1], w[g*4+2], w[g*4+3]);
    }

    // OC constants: dequant = act_scale * w_scale[oc] * fused_bn_scale[oc]
    // We precompute w_scale[oc] * fused_bn_scale[oc], multiply act_scale at runtime
    kd.oc_scale.resize(noc);
    kd.oc_bias.resize(noc);
    for (int o = 0; o < noc; o++) {
        kd.oc_scale[o] = w_scale[o] * layer.fused_scale[o];
        kd.oc_bias[o] = layer.fused_bias[o];
    }
    printf("  Shortcut %s: quantized FP32→INT8 (%d OC, %d IC)\n",
           layer.name.c_str(), noc, kd.col_rows);
}

// Ternary conv: uint8 input → im2col → repack → dpbusd GEMM
// If residual != nullptr: fuses add + relu + requant → outputs TensorU8 via out_u8
// If residual == nullptr: outputs FP32 via out_fp32
// Epilogue modes:
//   mode 0: output FP32 (out_fp32)
//   mode 1: fused ReLU + requant → uint8 (out_u8) — for conv1
//   mode 2: fused residual + ReLU + requant → uint8 (out_u8) — for conv2
static void conv2d_ternary_from_u8(const TensorU8& input, const FusedConv& layer,
                                    const I8KernelData& kd,
                                    uint8_t* col_u8, uint8_t* col_packed,
                                    int epilogue_mode,
                                    Tensor* out_fp32,
                                    TensorU8* out_u8, float out_u8_scale,
                                    const TensorU8* residual_u8 = nullptr,
                                    const Tensor* residual_fp32 = nullptr,
                                    float residual_scale = 0.0f,
                                    double* t_im2col = nullptr,
                                    double* t_gemm = nullptr) {
    auto& p = layer.p;
    int oh = (input.h + 2*p.padding - p.kernel_h) / p.stride + 1;
    int ow = (input.w + 2*p.padding - p.kernel_w) / p.stride + 1;

    if (epilogue_mode == 0) out_fp32->reshape(input.n, p.out_channels, oh, ow);
    else out_u8->reshape(input.n, p.out_channels, oh, ow, out_u8_scale);

    int col_rows = kd.col_rows;
    int col_cols = oh * ow;
    int col_rows_pad4 = kd.col_rows_pad4;
    int num_k_groups = col_rows_pad4 / 4;
    int cc8 = (col_cols + 7) & ~7;

    float act_scale = input.scale;

    for (int ni = 0; ni < input.n; ni++) {
        // ── im2col: parallel across K-rows ──
        auto _t0 = std::chrono::high_resolution_clock::now();
        {
            int ic = input.c, ih = input.h, iw = input.w;
            int kH = p.kernel_h, kW = p.kernel_w;
            int stride_v = p.stride, pad = p.padding;

            #pragma omp for schedule(static) nowait
            for (int k = 0; k < col_rows_pad4; k++) {
                uint8_t* dst = col_u8 + (size_t)k * col_cols;
                if (k >= col_rows) {
                    memset(dst, 0, col_cols);
                    continue;
                }
                int c = k / (kH * kW);
                int rem = k % (kH * kW);
                int kh = rem / kW, kw = rem % kW;
                const uint8_t* ch = input.ptr(ni, c);

                if (stride_v == 1 && pad == 1 && kH == 3 && kW == 3) {
                    // Specialized 3×3 stride-1 pad-1: memcpy per row
                    for (int oh_i = 0; oh_i < oh; oh_i++) {
                        int ih_i = oh_i - 1 + kh;
                        uint8_t* row_dst = dst + oh_i * ow;
                        if (ih_i < 0 || ih_i >= ih) {
                            memset(row_dst, 0, ow);
                        } else {
                            const uint8_t* src_row = ch + ih_i * iw;
                            if (kw == 0) {
                                row_dst[0] = 0;
                                memcpy(row_dst + 1, src_row, iw - 1);
                            } else if (kw == 1) {
                                memcpy(row_dst, src_row, iw);
                            } else {
                                memcpy(row_dst, src_row + 1, iw - 1);
                                row_dst[iw - 1] = 0;
                            }
                        }
                    }
                } else {
                    // Generic im2col row
                    int idx = 0;
                    for (int oh_i = 0; oh_i < oh; oh_i++) {
                        int row = oh_i * stride_v - pad + kh;
                        for (int ow_i = 0; ow_i < ow; ow_i++) {
                            int cp = ow_i * stride_v - pad + kw;
                            dst[idx++] = (row >= 0 && row < ih && cp >= 0 && cp < iw)
                                         ? ch[row * iw + cp] : 0;
                        }
                    }
                }
            }
        }
        #pragma omp barrier  // im2col must complete before repack

        // ── Repack: parallel across kg groups ──
        {
            int nkg = num_k_groups;
            int n_jg_full = col_cols / 8;
            int n_jg_total = cc8 / 8;

            #pragma omp for schedule(static)
            for (int kg = 0; kg < nkg; kg++) {
                const uint8_t* r0 = col_u8 + (kg*4+0)*col_cols;
                const uint8_t* r1 = col_u8 + (kg*4+1)*col_cols;
                const uint8_t* r2 = col_u8 + (kg*4+2)*col_cols;
                const uint8_t* r3 = col_u8 + (kg*4+3)*col_cols;

                int jg = 0;
                for (; jg < n_jg_full; jg++) {
                    int j = jg*8;
                    __m128i v0 = _mm_loadl_epi64((__m128i*)(r0+j));
                    __m128i v1 = _mm_loadl_epi64((__m128i*)(r1+j));
                    __m128i v2 = _mm_loadl_epi64((__m128i*)(r2+j));
                    __m128i v3 = _mm_loadl_epi64((__m128i*)(r3+j));
                    __m128i lo01 = _mm_unpacklo_epi8(v0, v1);
                    __m128i lo23 = _mm_unpacklo_epi8(v2, v3);
                    _mm_storeu_si128((__m128i*)(col_packed + (jg*nkg+kg)*32),
                                     _mm_unpacklo_epi16(lo01, lo23));
                    _mm_storeu_si128((__m128i*)(col_packed + (jg*nkg+kg)*32 + 16),
                                     _mm_unpackhi_epi16(lo01, lo23));
                }
                for (; jg < n_jg_total; jg++) {
                    uint8_t* out = col_packed + (jg*nkg+kg)*32;
                    for (int lane = 0; lane < 8; lane++) {
                        int j = jg*8+lane;
                        if (j < col_cols) {
                            out[lane*4+0] = r0[j]; out[lane*4+1] = r1[j];
                            out[lane*4+2] = r2[j]; out[lane*4+3] = r3[j];
                        } else {
                            *(uint32_t*)(out+lane*4) = 0;
                        }
                    }
                }
            }
        }
        // implicit barrier after omp for — repack done before GEMM

        #pragma omp single nowait
        {
            if (t_im2col) *t_im2col += std::chrono::duration<double,std::milli>(
                std::chrono::high_resolution_clock::now() - _t0).count();
        }

        // ── dpbusd GEMM: 8-OC register-tiled — distributed across thread team ──
        // 8 accumulators in YMM registers (no stack spills).
        // Register budget: 8 acc + 1 activation + 1 weight(reused) = 10 of 16 YMM.
        // 2× activation reuse vs 16-OC (was 2.7× worse with 6-OC).
        const int OC_TILE = 14;
        int n_jg = cc8 / 8;
        auto _tg0 = std::chrono::high_resolution_clock::now();

        #pragma omp for schedule(dynamic)
        for (int oc_base = 0; oc_base < (int)p.out_channels; oc_base += OC_TILE) {
            int oc_end = std::min(oc_base+OC_TILE, (int)p.out_channels);
            int n_oc = oc_end - oc_base;

            const int32_t* wpk_ptrs[14];
            float dequants[14], biases_arr[14];
            for (int t = 0; t < n_oc; t++) {
                int oc = oc_base+t;
                wpk_ptrs[t] = kd.weights_packed.data() + oc*num_k_groups;
                dequants[t] = act_scale * kd.oc_scale[oc];
                biases_arr[t] = kd.oc_bias[oc];
            }
            for (int t = n_oc; t < 14; t++) wpk_ptrs[t] = wpk_ptrs[0];

            for (int jg = 0; jg < n_jg; jg++) {
                __m256i r0 = _mm256_setzero_si256();
                __m256i r1 = _mm256_setzero_si256();
                __m256i r2 = _mm256_setzero_si256();
                __m256i r3 = _mm256_setzero_si256();
                __m256i r4 = _mm256_setzero_si256();
                __m256i r5 = _mm256_setzero_si256();
                __m256i r6 = _mm256_setzero_si256();
                __m256i r7 = _mm256_setzero_si256();
                __m256i r8 = _mm256_setzero_si256();
                __m256i r9 = _mm256_setzero_si256();
                __m256i r10 = _mm256_setzero_si256();
                __m256i r11 = _mm256_setzero_si256();
                __m256i r12 = _mm256_setzero_si256();
                __m256i r13 = _mm256_setzero_si256();

                for (int kg = 0; kg < num_k_groups; kg++) {
                    __m256i a = _mm256_loadu_si256(
                        (const __m256i*)(col_packed+(jg*num_k_groups+kg)*32));
                    __m256i w;

                    w = _mm256_set1_epi32(wpk_ptrs[0][kg]);
                    r0 = _mm256_dpbusd_epi32(r0, a, w);
                    w = _mm256_set1_epi32(wpk_ptrs[1][kg]);
                    r1 = _mm256_dpbusd_epi32(r1, a, w);
                    w = _mm256_set1_epi32(wpk_ptrs[2][kg]);
                    r2 = _mm256_dpbusd_epi32(r2, a, w);
                    w = _mm256_set1_epi32(wpk_ptrs[3][kg]);
                    r3 = _mm256_dpbusd_epi32(r3, a, w);
                    w = _mm256_set1_epi32(wpk_ptrs[4][kg]);
                    r4 = _mm256_dpbusd_epi32(r4, a, w);
                    w = _mm256_set1_epi32(wpk_ptrs[5][kg]);
                    r5 = _mm256_dpbusd_epi32(r5, a, w);
                    w = _mm256_set1_epi32(wpk_ptrs[6][kg]);
                    r6 = _mm256_dpbusd_epi32(r6, a, w);
                    w = _mm256_set1_epi32(wpk_ptrs[7][kg]);
                    r7 = _mm256_dpbusd_epi32(r7, a, w);
                    w = _mm256_set1_epi32(wpk_ptrs[8][kg]);
                    r8 = _mm256_dpbusd_epi32(r8, a, w);
                    w = _mm256_set1_epi32(wpk_ptrs[9][kg]);
                    r9 = _mm256_dpbusd_epi32(r9, a, w);
                    w = _mm256_set1_epi32(wpk_ptrs[10][kg]);
                    r10 = _mm256_dpbusd_epi32(r10, a, w);
                    w = _mm256_set1_epi32(wpk_ptrs[11][kg]);
                    r11 = _mm256_dpbusd_epi32(r11, a, w);
                    w = _mm256_set1_epi32(wpk_ptrs[12][kg]);
                    r12 = _mm256_dpbusd_epi32(r12, a, w);
                    w = _mm256_set1_epi32(wpk_ptrs[13][kg]);
                    r13 = _mm256_dpbusd_epi32(r13, a, w);
                }

                __m256i acc_arr[14] = {r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13};
                int j = jg*8;

                if (epilogue_mode == 0) {
                    for (int t = 0; t < n_oc; t++) {
                        float* dst = out_fp32->ptr(ni, oc_base+t);
                        __m256 result = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc_arr[t]),
                            _mm256_set1_ps(dequants[t]), _mm256_set1_ps(biases_arr[t]));
                        if (j+8 <= col_cols)
                            _mm256_storeu_ps(dst+j, result);
                        else {
                            float tmp[8]; _mm256_storeu_ps(tmp, result);
                            for (int jj = j; jj < col_cols; jj++) dst[jj] = tmp[jj-j];
                        }
                    }
                } else {
                    __m256 zero = _mm256_setzero_ps();
                    float inv_out = 1.0f / out_u8_scale;
                    __m256 inv_out_v = _mm256_set1_ps(inv_out);
                    __m256i lo_i = _mm256_setzero_si256(), hi_i = _mm256_set1_epi32(255);

                    for (int t = 0; t < n_oc; t++) {
                        int oc = oc_base + t;
                        __m256 val = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc_arr[t]),
                            _mm256_set1_ps(dequants[t]), _mm256_set1_ps(biases_arr[t]));

                        if (epilogue_mode == 2) {
                            if (residual_u8) {
                                __m256i ru8 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(residual_u8->ptr(ni, oc) + j)));
                                val = _mm256_add_ps(val, _mm256_mul_ps(_mm256_cvtepi32_ps(ru8), _mm256_set1_ps(residual_scale)));
                            } else {
                                val = _mm256_add_ps(val, _mm256_loadu_ps(residual_fp32->ptr(ni, oc) + j));
                            }
                        }

                        val = _mm256_max_ps(val, zero);
                        __m256i i32 = _mm256_cvtps_epi32(_mm256_mul_ps(val, inv_out_v));
                        i32 = _mm256_min_epi32(_mm256_max_epi32(i32, lo_i), hi_i);

                        uint8_t* dst = out_u8->ptr(ni, oc) + j;
                        if (j+8 <= col_cols) {
                            __m256i p16 = _mm256_packus_epi32(i32, i32);
                            p16 = _mm256_permute4x64_epi64(p16, 0xD8);
                            _mm_storel_epi64((__m128i*)dst,
                                _mm_packus_epi16(_mm256_castsi256_si128(p16), _mm256_castsi256_si128(p16)));
                        } else {
                            alignas(32) int32_t tmp[8];
                            _mm256_storeu_si256((__m256i*)tmp, i32);
                            for (int jj = j; jj < col_cols; jj++) dst[jj-j] = (uint8_t)tmp[jj-j];
                        }
                    }
                }
            }
        }
        if (t_gemm) {
            #pragma omp single
            { *t_gemm += std::chrono::duration<double,std::milli>(
                std::chrono::high_resolution_clock::now() - _tg0).count(); }
        }
    }
}

// ─── Utilities ──────────────────────────────────────────────

static void relu_inplace(Tensor& t) {
    int n = t.total(), i = 0;
    __m256 zero = _mm256_setzero_ps();
    for (; i+7 < n; i += 8)
        _mm256_storeu_ps(t.data.data()+i, _mm256_max_ps(zero, _mm256_loadu_ps(t.data.data()+i)));
    for (; i < n; i++) t.data[i] = std::max(0.0f, t.data[i]);
}

static void add_inplace(Tensor& a, const Tensor& b) {
    int n = a.total(), i = 0;
    for (; i+7 < n; i += 8)
        _mm256_storeu_ps(a.data.data()+i,
            _mm256_add_ps(_mm256_loadu_ps(a.data.data()+i), _mm256_loadu_ps(b.data.data()+i)));
    for (; i < n; i++) a.data[i] += b.data[i];
}

// Fused: conv2_fp32 + identity_u8 → add + ReLU + requant → uint8
// One pass, zero intermediate FP32 tensors for identity
static TensorU8 fused_add_relu_requant_u8(const Tensor& conv2_out, const TensorU8& identity,
                                            float out_scale) {
    TensorU8 out(conv2_out.n, conv2_out.c, conv2_out.h, conv2_out.w, out_scale);
    float id_scale = identity.scale;
    float inv_out = 1.0f / out_scale;
    int tot = conv2_out.total(), i = 0;

    __m256 id_sv = _mm256_set1_ps(id_scale);
    __m256 out_sv = _mm256_set1_ps(inv_out);
    __m256 zero = _mm256_setzero_ps();
    __m256i lo = _mm256_setzero_si256(), hi = _mm256_set1_epi32(255);

    for (; i+7 < tot; i += 8) {
        // Dequant identity uint8 → FP32
        __m256i id_u8 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(identity.data.data()+i)));
        __m256 id_fp = _mm256_mul_ps(_mm256_cvtepi32_ps(id_u8), id_sv);
        // Add conv2 output
        __m256 sum = _mm256_add_ps(_mm256_loadu_ps(conv2_out.data.data()+i), id_fp);
        // ReLU
        sum = _mm256_max_ps(sum, zero);
        // Requant to uint8
        __m256i i32 = _mm256_cvtps_epi32(_mm256_mul_ps(sum, out_sv));
        i32 = _mm256_min_epi32(_mm256_max_epi32(i32, lo), hi);
        __m256i p16 = _mm256_packus_epi32(i32, i32);
        p16 = _mm256_permute4x64_epi64(p16, 0xD8);
        __m128i p8 = _mm_packus_epi16(_mm256_castsi256_si128(p16), _mm256_castsi256_si128(p16));
        _mm_storel_epi64((__m128i*)(out.data.data()+i), p8);
    }
    for (; i < tot; i++) {
        float v = conv2_out.data[i] + identity.data[i] * id_scale;
        v = std::max(0.0f, v);
        int q = (int)roundf(v * inv_out);
        out.data[i] = (uint8_t)std::max(0, std::min(255, q));
    }
    return out;
}

// Fused: conv2_fp32 + shortcut_fp32 → add + ReLU + requant → uint8
static TensorU8 fused_add_relu_requant_fp(const Tensor& conv2_out, const Tensor& shortcut,
                                            float out_scale) {
    TensorU8 out(conv2_out.n, conv2_out.c, conv2_out.h, conv2_out.w, out_scale);
    float inv_out = 1.0f / out_scale;
    int tot = conv2_out.total(), i = 0;

    __m256 out_sv = _mm256_set1_ps(inv_out);
    __m256 zero = _mm256_setzero_ps();
    __m256i lo = _mm256_setzero_si256(), hi = _mm256_set1_epi32(255);

    for (; i+7 < tot; i += 8) {
        __m256 sum = _mm256_add_ps(_mm256_loadu_ps(conv2_out.data.data()+i),
                                    _mm256_loadu_ps(shortcut.data.data()+i));
        sum = _mm256_max_ps(sum, zero);
        __m256i i32 = _mm256_cvtps_epi32(_mm256_mul_ps(sum, out_sv));
        i32 = _mm256_min_epi32(_mm256_max_epi32(i32, lo), hi);
        __m256i p16 = _mm256_packus_epi32(i32, i32);
        p16 = _mm256_permute4x64_epi64(p16, 0xD8);
        __m128i p8 = _mm_packus_epi16(_mm256_castsi256_si128(p16), _mm256_castsi256_si128(p16));
        _mm_storel_epi64((__m128i*)(out.data.data()+i), p8);
    }
    for (; i < tot; i++) {
        float v = std::max(0.0f, conv2_out.data[i] + shortcut.data[i]);
        out.data[i] = (uint8_t)std::max(0, std::min(255, (int)roundf(v * inv_out)));
    }
    return out;
}

static Tensor avgpool_global(const Tensor& input) {
    Tensor output(input.n, input.c, 1, 1);
    int spatial = input.h * input.w;
    float inv = 1.0f / spatial;
    for (int ni = 0; ni < input.n; ni++)
        for (int ci = 0; ci < input.c; ci++) {
            const float* p = input.ptr(ni, ci);
            __m256 acc = _mm256_setzero_ps();
            int i = 0;
            for (; i+7 < spatial; i += 8) acc = _mm256_add_ps(acc, _mm256_loadu_ps(p+i));
            __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
            h = _mm_add_ps(h, _mm_shuffle_ps(h, h, _MM_SHUFFLE(2,3,0,1)));
            h = _mm_add_ps(h, _mm_shuffle_ps(h, h, _MM_SHUFFLE(1,0,3,2)));
            float sum = _mm_cvtss_f32(h);
            for (; i < spatial; i++) sum += p[i];
            output.data[ni*input.c+ci] = sum*inv;
        }
    return output;
}

// Avgpool from uint8 → FP32 (dequantize on the fly)
static Tensor avgpool_global_u8(const TensorU8& input) {
    Tensor output(input.n, input.c, 1, 1);
    int spatial = input.h * input.w;
    float inv = input.scale / spatial;
    for (int ni = 0; ni < input.n; ni++)
        for (int ci = 0; ci < input.c; ci++) {
            const uint8_t* p = input.ptr(ni, ci);
            int32_t sum = 0;
            for (int i = 0; i < spatial; i++) sum += p[i];
            output.data[ni*input.c+ci] = sum * inv;
        }
    return output;
}

static std::vector<float> fc_forward(const Tensor& input, const FC& fc) {
    int in_f = fc.in_features, out_f = fc.out_features;
    std::vector<float> output(out_f);
    for (int o = 0; o < out_f; o++) {
        const float* w = fc.weight.data() + o*in_f;
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i+7 < in_f; i += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(w+i), _mm256_loadu_ps(input.data.data()+i), acc);
        __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        h = _mm_add_ps(h, _mm_shuffle_ps(h, h, _MM_SHUFFLE(2,3,0,1)));
        h = _mm_add_ps(h, _mm_shuffle_ps(h, h, _MM_SHUFFLE(1,0,3,2)));
        float sum = _mm_cvtss_f32(h) + fc.bias[o];
        for (; i < in_f; i++) sum += w[i] * input.data[i];
        output[o] = sum;
    }
    return output;
}

// ─── ResNet-18 INT8-Resident Forward Pass ───────────────────

struct BasicBlockIndices { int conv1, conv2, shortcut; };

// Static scales loaded from calibration
struct LayerScales {
    float conv1_relu_out;    // post conv1+BN+ReLU → uint8 scale for next conv input
    float conv2_bn_out;      // post conv2+BN → signed, for residual add
    float shortcut_out;      // post shortcut+BN → signed, for residual add
    float block_relu_out;    // post add+ReLU → uint8 scale for next block input
};

class Engine {
public:
    Model model;
    std::vector<BasicBlockIndices> blocks;
    std::vector<I8KernelData> i8_kernels;  // for both ternary and shortcut convs

    // Static scales per block
    float initial_relu_scale;
    std::vector<LayerScales> block_scales;

    // Preallocated workspace — allocated once, reused across all conv calls
    std::vector<uint8_t> ws_col_u8;
    std::vector<uint8_t> ws_col_packed;

    // Preallocated activation buffers — zero malloc on hot path
    Tensor ws_fp32_a, ws_fp32_b;   // conv1 output + shortcut
    TensorU8 ws_u8_a, ws_u8_b, ws_u8_c;  // ping-pong activations

    void init(const Model& m, const char* scales_path) {
        model = m;
        i8_kernels.resize(model.convs.size());
        for (size_t i = 0; i < model.convs.size(); i++) {
            if (model.convs[i].is_ternary)
                prepare_ternary_kernel(model.convs[i], i8_kernels[i]);
            else if (i > 0)  // skip first conv (stays FP32 on FP32 input), prepare shortcuts
                prepare_shortcut_kernel(model.convs[i], i8_kernels[i]);
        }

        int idx = 1;
        bool has_sc[] = {false, true, true, true};
        for (int l = 0; l < 4; l++)
            for (int b = 0; b < 2; b++) {
                BasicBlockIndices bi;
                bi.conv1 = idx++; bi.conv2 = idx++;
                bi.shortcut = (b==0 && has_sc[l]) ? idx++ : -1;
                blocks.push_back(bi);
            }

        // Load static scales from JSON
        load_scales(scales_path);

        // Preallocate workspace with max size across all ternary conv layers
        size_t max_col_u8 = 0, max_col_packed = 0;
        for (size_t i = 0; i < model.convs.size(); i++) {
            if (i == 0) continue;  // first conv is FP32 on FP32 input
            auto& p = model.convs[i].p;
            auto& kd = i8_kernels[i];
            // Compute output spatial size for this layer
            // We need the input size, which varies — use worst case
            // For ResNet-18 CIFAR-10, max spatial is 32×32=1024 for layer1
            // max col_rows is 512*3*3=4608 for layer4
            int cr_pad4 = kd.col_rows_pad4;
            int nkg = cr_pad4 / 4;
            // Max spatial: conservatively use 1024 (32×32)
            int max_spatial = 1024;
            int cc8 = (max_spatial + 7) & ~7;
            size_t cu = (size_t)cr_pad4 * max_spatial;
            size_t cp = (size_t)(cc8/8) * nkg * 32;
            if (cu > max_col_u8) max_col_u8 = cu;
            if (cp > max_col_packed) max_col_packed = cp;
        }
        ws_col_u8.resize(max_col_u8);
        ws_col_packed.resize(max_col_packed);
        printf("Workspace: col_u8=%zuKB col_packed=%zuKB (preallocated once)\n",
               max_col_u8/1024, max_col_packed/1024);

        // Preallocate activation buffers at max size (64*32*32 = 64KB u8, 256KB fp32)
        ws_fp32_a = Tensor(1, 64, 32, 32);
        ws_fp32_b = Tensor(1, 64, 32, 32);
        ws_u8_a = TensorU8(1, 64, 32, 32, 1.0f);
        ws_u8_b = TensorU8(1, 64, 32, 32, 1.0f);
        ws_u8_c = TensorU8(1, 64, 32, 32, 1.0f);

        printf("Engine: %zu convs, %zu blocks, INT8-resident + dpbusd\n",
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

    // Profiling counters (accumulated across runs, divide by run count)
    double prof_conv1_fp32 = 0, prof_ternary_im2col = 0, prof_ternary_gemm = 0;
    double prof_epilogue = 0, prof_pool_fc = 0;
    int prof_runs = 0;

    void print_profile(int nruns) {
        double total = prof_conv1_fp32 + prof_ternary_im2col + prof_ternary_gemm + prof_pool_fc;
        printf("\n═══════════════════════════════════════════════════\n");
        printf("ENGINE PROFILE (%d runs, 1T per-inference avg)\n", nruns);
        printf("═══════════════════════════════════════════════════\n");
        printf("  Conv1 FP32+ReLU+quant:  %8.3f ms  (%4.1f%%)\n", prof_conv1_fp32/nruns, 100*prof_conv1_fp32/total);
        printf("  Ternary im2col+repack:  %8.3f ms  (%4.1f%%)\n", prof_ternary_im2col/nruns, 100*prof_ternary_im2col/total);
        printf("  Ternary GEMM+epilogue:  %8.3f ms  (%4.1f%%)\n", prof_ternary_gemm/nruns, 100*prof_ternary_gemm/total);
        printf("  Pool+FC:                %8.3f ms  (%4.1f%%)\n", prof_pool_fc/nruns, 100*prof_pool_fc/total);
        printf("  ─────────────────────────────────────────────\n");
        printf("  TOTAL:                  %8.3f ms\n", total/nruns);
        printf("═══════════════════════════════════════════════════\n");
    }
    void reset_profile() { prof_conv1_fp32=prof_ternary_im2col=prof_ternary_gemm=prof_pool_fc=0; prof_runs=0; }

    std::vector<float> forward(const Tensor& input) {
        auto tnow = []{ return std::chrono::high_resolution_clock::now(); };
        auto tms = [](auto a, auto b){ return std::chrono::duration<double,std::milli>(b-a).count(); };

        TensorU8* final_x = nullptr;

        #pragma omp parallel if(omp_get_max_threads() > 1)
        {
            // ── Conv1 FP32: im2col (single) → GEMM+ReLU+quant (parallel over OC) ──
            auto t0 = tnow();
            {
                auto& p0 = model.convs[0].p;
                int c1_oh = (input.h + 2*p0.padding - p0.kernel_h) / p0.stride + 1;
                int c1_ow = (input.w + 2*p0.padding - p0.kernel_w) / p0.stride + 1;
                int c1_cr = p0.in_channels * p0.kernel_h * p0.kernel_w;
                int c1_cc = c1_oh * c1_ow;

                #pragma omp single
                {
                    ws_fp32_a.reshape(input.n, p0.out_channels, c1_oh, c1_ow);
                    ws_u8_a.reshape(input.n, p0.out_channels, c1_oh, c1_ow, initial_relu_scale);
                    // im2col into preallocated workspace
                    im2col_fp32(input, 0, p0.kernel_h, p0.kernel_w,
                                p0.stride, p0.padding, c1_oh, c1_ow,
                                (float*)ws_col_u8.data());  // reuse u8 workspace for fp32 im2col
                }

                // Fused GEMM + ReLU + quantize — parallel over OC
                float inv_scale = 1.0f / initial_relu_scale;
                const float* col_fp = (const float*)ws_col_u8.data();

                #pragma omp for schedule(static)
                for (int oc = 0; oc < (int)p0.out_channels; oc++) {
                    float* out = ws_fp32_a.ptr(0, oc);
                    const float* w = model.convs[0].weights_fp32.data() + oc * c1_cr;
                    float bias = model.convs[0].fused_bias[oc];

                    // GEMM: out = W × col + bias
                    memset(out, 0, c1_cc * sizeof(float));
                    for (int k = 0; k < c1_cr; k++) {
                        const float* src = col_fp + k * c1_cc;
                        __m256 wv = _mm256_set1_ps(w[k]);
                        int j = 0;
                        for (; j + 7 < c1_cc; j += 8)
                            _mm256_storeu_ps(out+j, _mm256_fmadd_ps(wv, _mm256_loadu_ps(src+j), _mm256_loadu_ps(out+j)));
                        for (; j < c1_cc; j++) out[j] += w[k] * src[j];
                    }

                    // Fused: + bias → ReLU → quantize to u8
                    __m256 bv = _mm256_set1_ps(bias);
                    __m256 zero = _mm256_setzero_ps();
                    __m256 inv_v = _mm256_set1_ps(inv_scale);
                    __m256i lo_i = _mm256_setzero_si256(), hi_i = _mm256_set1_epi32(255);
                    uint8_t* dst_u8 = ws_u8_a.ptr(0, oc);
                    int j = 0;
                    for (; j + 7 < c1_cc; j += 8) {
                        __m256 v = _mm256_add_ps(_mm256_loadu_ps(out+j), bv);
                        v = _mm256_max_ps(v, zero);  // ReLU
                        _mm256_storeu_ps(out+j, v);   // store FP32 (needed? only if ws_fp32_a used later)
                        __m256i i32 = _mm256_cvtps_epi32(_mm256_mul_ps(v, inv_v));
                        i32 = _mm256_min_epi32(_mm256_max_epi32(i32, lo_i), hi_i);
                        __m256i p16 = _mm256_packus_epi32(i32, i32);
                        p16 = _mm256_permute4x64_epi64(p16, 0xD8);
                        _mm_storel_epi64((__m128i*)(dst_u8+j),
                            _mm_packus_epi16(_mm256_castsi256_si128(p16), _mm256_castsi256_si128(p16)));
                    }
                    for (; j < c1_cc; j++) {
                        float v = std::max(0.0f, out[j] + bias);
                        out[j] = v;
                        dst_u8[j] = (uint8_t)std::max(0, std::min(255, (int)roundf(v * inv_scale)));
                    }
                }
            }
            #pragma omp single nowait
            { prof_conv1_fp32 += tms(t0, tnow()); }

            // Ping-pong: each thread has local copies of the pointers
            // (all identical, pointing to the same Engine-owned buffers)
            TensorU8* x = &ws_u8_a;
            TensorU8* out1 = &ws_u8_b;
            TensorU8* out2 = &ws_u8_c;

            for (size_t bi = 0; bi < blocks.size(); bi++) {
                auto& blk = blocks[bi];
                auto& sc = block_scales[bi];

                // Conv1: all threads enter — single does im2col, for does GEMM
                conv2d_ternary_from_u8(*x, model.convs[blk.conv1], i8_kernels[blk.conv1],
                                       ws_col_u8.data(), ws_col_packed.data(),
                                       1, nullptr, out1, sc.conv1_relu_out,
                                       nullptr, nullptr, 0.0f,
                                       &prof_ternary_im2col, &prof_ternary_gemm);

                if (blk.shortcut >= 0) {
                    conv2d_ternary_from_u8(*x, model.convs[blk.shortcut], i8_kernels[blk.shortcut],
                                           ws_col_u8.data(), ws_col_packed.data(),
                                           0, &ws_fp32_b, out2, 0.0f,
                                           nullptr, nullptr, 0.0f,
                                           &prof_ternary_im2col, &prof_ternary_gemm);
                    conv2d_ternary_from_u8(*out1, model.convs[blk.conv2], i8_kernels[blk.conv2],
                                           ws_col_u8.data(), ws_col_packed.data(),
                                           2, nullptr, out2, sc.block_relu_out,
                                           nullptr, &ws_fp32_b, 0.0f,
                                           &prof_ternary_im2col, &prof_ternary_gemm);
                } else {
                    conv2d_ternary_from_u8(*out1, model.convs[blk.conv2], i8_kernels[blk.conv2],
                                           ws_col_u8.data(), ws_col_packed.data(),
                                           2, nullptr, out2, sc.block_relu_out,
                                           x, nullptr, x->scale,
                                           &prof_ternary_im2col, &prof_ternary_gemm);
                }
                TensorU8* tmp = x;
                x = out2;
                out2 = tmp;
            }

            // All threads agree on final x (same pointer rotation)
            #pragma omp single
            { final_x = x; }
        } // end omp parallel — threads go idle here, once per inference

        auto tp = tnow();
        Tensor pooled = avgpool_global_u8(*final_x);
        auto result = fc_forward(pooled, model.fc);
        prof_pool_fc += tms(tp, tnow());
        prof_runs++;
        return result;
    }
};
