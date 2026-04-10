#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <immintrin.h>

// Isolated single-method benchmark for perf stat comparison.
// Usage: bench_lut_perf <method> where method = dpbusd | vpshufb
// Runs layer2 (128 OC, 256 spatial, 1152 reduction) × 5000 iterations.

void gemm_dpbusd(const uint8_t* act_packed, const int32_t* wt_packed,
                 int32_t* out, int OC, int N, int K) {
    int nkg = K / 4, n_jg = N / 8;
    for (int oc = 0; oc < OC; oc++) {
        const int32_t* wp = wt_packed + oc * nkg;
        for (int jg = 0; jg < n_jg; jg++) {
            __m256i acc = _mm256_setzero_si256();
            for (int kg = 0; kg < nkg; kg++) {
                __m256i a = _mm256_loadu_si256((__m256i*)(act_packed + (kg*n_jg+jg)*32));
                __m256i w = _mm256_set1_epi32(wp[kg]);
                acc = _mm256_dpbusd_epi32(acc, a, w);
            }
            _mm256_storeu_si256((__m256i*)(out + oc*N + jg*8), acc);
        }
    }
}

struct NibbleLUT {
    std::vector<__m256i> tables;
    void build(const int8_t* weights, int OC, int K) {
        tables.resize(OC * K);
        for (int oc = 0; oc < OC; oc++) {
            for (int k = 0; k < K; k++) {
                int8_t w = weights[oc*K+k];
                alignas(16) int8_t lo[16], hi[16];
                for (int i = 0; i < 16; i++) {
                    lo[i] = (int8_t)(w * i);
                    int16_t hv = (int16_t)(w * i * 16);
                    hi[i] = (int8_t)(hv & 0xFF);
                }
                tables[oc*K+k] = _mm256_set_m128i(
                    _mm_loadu_si128((__m128i*)hi),
                    _mm_loadu_si128((__m128i*)lo));
            }
        }
    }
};

void gemm_vpshufb(const uint8_t* act, const NibbleLUT& lut,
                  int32_t* out, int OC, int N, int K) {
    __m256i mask_lo = _mm256_set1_epi8(0x0F);
    for (int oc = 0; oc < OC; oc++) {
        for (int n = 0; n < N; n += 32) {
            __m256i acc_lo = _mm256_setzero_si256();
            __m256i acc_hi = _mm256_setzero_si256();
            for (int k = 0; k < K; k++) {
                __m256i a = _mm256_loadu_si256((__m256i*)(act + k*N + n));
                __m256i lo_nib = _mm256_and_si256(a, mask_lo);
                __m256i hi_nib = _mm256_and_si256(_mm256_srli_epi16(a, 4), mask_lo);
                __m256i tbl = lut.tables[oc*K+k];
                __m256i tbl_lo = _mm256_broadcastsi128_si256(_mm256_castsi256_si128(tbl));
                __m256i tbl_hi = _mm256_broadcastsi128_si256(_mm256_extracti128_si256(tbl, 1));
                __m256i val_lo = _mm256_shuffle_epi8(tbl_lo, lo_nib);
                __m256i val_hi = _mm256_shuffle_epi8(tbl_hi, hi_nib);
                __m256i lo16_0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(val_lo));
                __m256i lo16_1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(val_lo, 1));
                __m256i hi16_0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(val_hi));
                __m256i hi16_1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(val_hi, 1));
                acc_lo = _mm256_add_epi16(acc_lo, _mm256_add_epi16(lo16_0, hi16_0));
                acc_hi = _mm256_add_epi16(acc_hi, _mm256_add_epi16(lo16_1, hi16_1));
            }
            _mm256_storeu_si256((__m256i*)(out+oc*N+n),    _mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc_lo)));
            _mm256_storeu_si256((__m256i*)(out+oc*N+n+8),  _mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc_lo,1)));
            _mm256_storeu_si256((__m256i*)(out+oc*N+n+16), _mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc_hi)));
            _mm256_storeu_si256((__m256i*)(out+oc*N+n+24), _mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc_hi,1)));
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <dpbusd|vpshufb>\n", argv[0]); return 1; }
    bool use_dpbusd = (strcmp(argv[1], "dpbusd") == 0);

    // Layer 2: 128 OC, 256 spatial (16×16), 1152 reduction (128 IC × 3 × 3)
    int OC = 128, N = 256, K = 1152;
    int nkg = K/4, n_jg = N/8;
    int N32 = (N+31)&~31;
    int iters = 5000;

    if (use_dpbusd) {
        std::vector<uint8_t> act(nkg*n_jg*32);
        std::vector<int32_t> wt(OC*nkg);
        std::vector<int32_t> out(OC*N);
        for (auto& v : act) v = rand()%256;
        for (auto& v : wt) v = rand();
        // warmup
        for (int i = 0; i < 200; i++) gemm_dpbusd(act.data(), wt.data(), out.data(), OC, N, K);
        // measured
        for (int i = 0; i < iters; i++) gemm_dpbusd(act.data(), wt.data(), out.data(), OC, N, K);
    } else {
        std::vector<uint8_t> act(K*N32);
        std::vector<int8_t> wt(OC*K);
        std::vector<int32_t> out(OC*N32);
        for (auto& v : act) v = rand()%256;
        for (auto& v : wt) v = (rand()%3)-1;
        NibbleLUT lut; lut.build(wt.data(), OC, K);
        for (int i = 0; i < 200; i++) gemm_vpshufb(act.data(), lut, out.data(), OC, N32, K);
        for (int i = 0; i < iters; i++) gemm_vpshufb(act.data(), lut, out.data(), OC, N32, K);
    }
    printf("Done: %s × %d iters\n", argv[1], iters);
    return 0;
}
