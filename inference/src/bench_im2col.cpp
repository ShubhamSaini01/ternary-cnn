// Benchmark: baseline two-pass vs fused im2col+repack vs unrolled im2col
// Reports per-invocation timing for each approach at layer1 dimensions.
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <immintrin.h>
#include "engine.h"

int main(int argc, char** argv) {
    // Layer1 dimensions: IC=64, 3x3, s1p1, input 32x32, output 32x32
    int ic = 64, ih = 32, iw = 32, oh = 32, ow = 32;
    int kH = 3, kW = 3, stride = 1, padding = 1;
    int col_rows = ic * kH * kW;  // 576
    int col_cols = oh * ow;       // 1024
    int col_rows_pad4 = (col_rows + 3) & ~3;  // 576
    int num_k_groups = col_rows_pad4 / 4;      // 144
    int cc8 = (col_cols + 7) & ~7;             // 1024
    int spatial = oh * ow;

    printf("Layer1 dimensions: IC=%d, 3x3, s1p1, spatial=%d\n", ic, spatial);
    printf("col_u8: [%d x %d] = %d KB\n", col_rows_pad4, col_cols, col_rows_pad4*col_cols/1024);
    printf("col_packed: [%d x %d x 32] = %d KB\n\n", cc8/8, num_k_groups, (cc8/8)*num_k_groups*32/1024);

    // Create fake uint8 input
    TensorU8 input(1, ic, ih, iw, 0.01f);
    for (size_t i = 0; i < input.data.size(); i++) input.data[i] = rand() % 128;

    std::vector<uint8_t> col_u8(col_rows_pad4 * col_cols);
    std::vector<uint8_t> col_packed((cc8/8) * num_k_groups * 32);

    int warmup = 500, runs = 5000;

    // ═══ APPROACH 1: Baseline two-pass (im2col + SSE repack) ═══
    auto run_baseline = [&]() {
        im2col_u8_3x3s1p1(input, 0, oh, ow, col_u8.data());
        for (int k = col_rows; k < col_rows_pad4; k++)
            memset(col_u8.data() + k*col_cols, 0, col_cols);
        int nkg = num_k_groups;
        for (int kg = 0; kg < nkg; kg++) {
            const uint8_t* r0 = col_u8.data() + (kg*4+0)*col_cols;
            const uint8_t* r1 = col_u8.data() + (kg*4+1)*col_cols;
            const uint8_t* r2 = col_u8.data() + (kg*4+2)*col_cols;
            const uint8_t* r3 = col_u8.data() + (kg*4+3)*col_cols;
            int jg = 0;
            for (; jg < col_cols/8; jg++) {
                int j = jg*8;
                __m128i v0 = _mm_loadl_epi64((__m128i*)(r0+j));
                __m128i v1 = _mm_loadl_epi64((__m128i*)(r1+j));
                __m128i v2 = _mm_loadl_epi64((__m128i*)(r2+j));
                __m128i v3 = _mm_loadl_epi64((__m128i*)(r3+j));
                __m128i lo01 = _mm_unpacklo_epi8(v0, v1);
                __m128i lo23 = _mm_unpacklo_epi8(v2, v3);
                _mm_storeu_si128((__m128i*)(col_packed.data() + (jg*nkg+kg)*32),
                                 _mm_unpacklo_epi16(lo01, lo23));
                _mm_storeu_si128((__m128i*)(col_packed.data() + (jg*nkg+kg)*32 + 16),
                                 _mm_unpackhi_epi16(lo01, lo23));
            }
        }
    };

    // ═══ APPROACH 2: Fused im2col→col_packed ═══
    auto run_fused = [&]() {
        fused_im2col_repack_3x3s1p1(input, 0, oh, ow, col_rows_pad4, col_packed.data());
    };

    // ═══ APPROACH 3: Unrolled 9-pos im2col + baseline repack ═══
    // Inline copy helpers
    auto copy_row = [](uint8_t* dst, const uint8_t* src, int n) {
        int i = 0;
        for (; i + 31 < n; i += 32)
            _mm256_storeu_si256((__m256i*)(dst+i), _mm256_loadu_si256((__m256i*)(src+i)));
        for (; i + 15 < n; i += 16)
            _mm_storeu_si128((__m128i*)(dst+i), _mm_loadu_si128((__m128i*)(src+i)));
        for (; i < n; i++) dst[i] = src[i];
    };
    auto zero_row = [](uint8_t* dst, int n) {
        int i = 0;
        __m256i z = _mm256_setzero_si256();
        for (; i + 31 < n; i += 32) _mm256_storeu_si256((__m256i*)(dst+i), z);
        for (; i < n; i++) dst[i] = 0;
    };
    auto im2col_pos = [&](const uint8_t* ch, int kh, int kw, uint8_t* col) {
        for (int oh_i = 0; oh_i < oh; oh_i++) {
            int ih_i = oh_i - 1 + kh;
            uint8_t* dst = col + oh_i * ow;
            if (ih_i < 0 || ih_i >= ih) { zero_row(dst, ow); }
            else {
                const uint8_t* src = ch + ih_i * iw;
                if (kw == 0) { dst[0] = 0; copy_row(dst+1, src, iw-1); }
                else if (kw == 1) { copy_row(dst, src, iw); }
                else { copy_row(dst, src+1, iw-1); dst[iw-1] = 0; }
            }
        }
    };
    auto run_unrolled = [&]() {
        uint8_t* col = col_u8.data();
        for (int c = 0; c < ic; c++) {
            const uint8_t* ch = input.ptr(0, c);
            im2col_pos(ch, 0, 0, col + 0*spatial);
            im2col_pos(ch, 0, 1, col + 1*spatial);
            im2col_pos(ch, 0, 2, col + 2*spatial);
            im2col_pos(ch, 1, 0, col + 3*spatial);
            im2col_pos(ch, 1, 1, col + 4*spatial);
            im2col_pos(ch, 1, 2, col + 5*spatial);
            im2col_pos(ch, 2, 0, col + 6*spatial);
            im2col_pos(ch, 2, 1, col + 7*spatial);
            im2col_pos(ch, 2, 2, col + 8*spatial);
            col += 9 * spatial;
        }
        for (int k = col_rows; k < col_rows_pad4; k++)
            memset(col_u8.data() + k*col_cols, 0, col_cols);
        // Same repack as baseline
        int nkg = num_k_groups;
        for (int kg = 0; kg < nkg; kg++) {
            const uint8_t* r0 = col_u8.data() + (kg*4+0)*col_cols;
            const uint8_t* r1 = col_u8.data() + (kg*4+1)*col_cols;
            const uint8_t* r2 = col_u8.data() + (kg*4+2)*col_cols;
            const uint8_t* r3 = col_u8.data() + (kg*4+3)*col_cols;
            for (int jg = 0; jg < col_cols/8; jg++) {
                int j = jg*8;
                __m128i v0 = _mm_loadl_epi64((__m128i*)(r0+j));
                __m128i v1 = _mm_loadl_epi64((__m128i*)(r1+j));
                __m128i v2 = _mm_loadl_epi64((__m128i*)(r2+j));
                __m128i v3 = _mm_loadl_epi64((__m128i*)(r3+j));
                __m128i lo01 = _mm_unpacklo_epi8(v0, v1);
                __m128i lo23 = _mm_unpacklo_epi8(v2, v3);
                _mm_storeu_si128((__m128i*)(col_packed.data() + (jg*nkg+kg)*32),
                                 _mm_unpacklo_epi16(lo01, lo23));
                _mm_storeu_si128((__m128i*)(col_packed.data() + (jg*nkg+kg)*32 + 16),
                                 _mm_unpackhi_epi16(lo01, lo23));
            }
        }
    };

    // ═══ Benchmark each ═══
    auto bench = [&](const char* name, auto fn) {
        for (int i = 0; i < warmup; i++) fn();
        std::vector<double> times(runs);
        for (int i = 0; i < runs; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            auto t1 = std::chrono::high_resolution_clock::now();
            times[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
        }
        std::sort(times.begin(), times.end());
        double med = times[runs/2];
        double p5 = times[(int)(runs*0.05)];
        double p95 = times[(int)(runs*0.95)];
        printf("%-45s  median=%6.1f us  p5=%6.1f  p95=%6.1f\n", name, med, p5, p95);
    };

    int mode = (argc > 1) ? atoi(argv[1]) : 0;
    if (mode == 0 || mode == 1) bench("1. Baseline: memcpy im2col + SSE repack", run_baseline);
    if (mode == 0 || mode == 2) bench("2. Fused: input -> col_packed (skip col_u8)", run_fused);
    if (mode == 0 || mode == 3) bench("3. Unrolled 9-pos + inline AVX2 copies", run_unrolled);

    return 0;
}
