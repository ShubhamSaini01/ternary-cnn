#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <algorithm>
#include <immintrin.h>

// Benchmark: dpbusd GEMM vs LUT for ternary CNN conv layers
// Tests on realistic CIFAR-10 ResNet-18 layer sizes

static inline double now_ms() {
    return std::chrono::duration<double,std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

// ── dpbusd spatial-packed GEMM (production kernel style) ──
// Processes 8 spatial positions × 4 reduction per dpbusd instruction
void gemm_dpbusd(const uint8_t* act_packed,   // [K/4][N/8][32] spatial-packed
                 const int32_t* wt_packed,     // [OC][K/4] broadcast-ready
                 int32_t* out,                 // [OC][N]
                 int OC, int N, int K) {
    int nkg = K / 4;
    int n_jg = N / 8;
    for (int oc = 0; oc < OC; oc++) {
        const int32_t* wp = wt_packed + oc * nkg;
        for (int jg = 0; jg < n_jg; jg++) {
            __m256i acc = _mm256_setzero_si256();
            for (int kg = 0; kg < nkg; kg++) {
                __m256i a = _mm256_loadu_si256((__m256i*)(act_packed + (kg * n_jg + jg) * 32));
                __m256i w = _mm256_set1_epi32(wp[kg]);
                acc = _mm256_dpbusd_epi32(acc, a, w);
            }
            _mm256_storeu_si256((__m256i*)(out + oc * N + jg * 8), acc);
        }
    }
}

// ── LUT approach (TL2-style) ──
// For each weight group of 4 values, precompute table[256] = dot(act_val, w_group)
// Then inference = table lookup per activation byte
void build_lut(const int8_t* weights, float* lut, int OC, int K) {
    // For each OC and each group of 4 weights, build 256-entry table
    // lut[oc][kg][256] where kg = K/4
    int nkg = K / 4;
    for (int oc = 0; oc < OC; oc++) {
        for (int kg = 0; kg < nkg; kg++) {
            float* tab = lut + (oc * nkg + kg) * 256;
            int8_t w0 = weights[(oc * K) + kg * 4 + 0];
            int8_t w1 = weights[(oc * K) + kg * 4 + 1];
            int8_t w2 = weights[(oc * K) + kg * 4 + 2];
            int8_t w3 = weights[(oc * K) + kg * 4 + 3];
            for (int v = 0; v < 256; v++) {
                // Each activation byte in spatial-packed layout holds one uint8 value
                tab[v] = (float)((int)v * (int)w0 + (int)v * (int)w1 +
                                 (int)v * (int)w2 + (int)v * (int)w3);
            }
        }
    }
}

// LUT inference: scalar version (1 lookup per activation byte)
void gemm_lut_scalar(const uint8_t* activations, // [N][K] row-major
                     const float* lut,            // [OC][K/4][256]
                     float* out,                  // [OC][N]
                     int OC, int N, int K) {
    int nkg = K / 4;
    for (int oc = 0; oc < OC; oc++) {
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int kg = 0; kg < nkg; kg++) {
                const float* tab = lut + (oc * nkg + kg) * 256;
                // Look up each of 4 activation bytes in this group
                sum += tab[activations[n * K + kg * 4 + 0]];
                sum += tab[activations[n * K + kg * 4 + 1]];
                sum += tab[activations[n * K + kg * 4 + 2]];
                sum += tab[activations[n * K + kg * 4 + 3]];
            }
            out[oc * N + n] = sum;
        }
    }
}

// LUT inference: gather version (8 lookups per gather instruction)
void gemm_lut_gather(const uint8_t* activations, // [N][K] row-major
                     const float* lut,            // [OC][K/4][256]
                     float* out,                  // [OC][N]
                     int OC, int N, int K) {
    int nkg = K / 4;
    for (int oc = 0; oc < OC; oc++) {
        for (int n = 0; n < N; n += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int kg = 0; kg < nkg; kg++) {
                const float* tab = lut + (oc * nkg + kg) * 256;
                // Gather 8 lookups for 8 spatial positions, byte 0
                for (int b = 0; b < 4; b++) {
                    __m256i idx = _mm256_set_epi32(
                        activations[(n+7)*K + kg*4+b], activations[(n+6)*K + kg*4+b],
                        activations[(n+5)*K + kg*4+b], activations[(n+4)*K + kg*4+b],
                        activations[(n+3)*K + kg*4+b], activations[(n+2)*K + kg*4+b],
                        activations[(n+1)*K + kg*4+b], activations[(n+0)*K + kg*4+b]);
                    __m256 val = _mm256_i32gather_ps(tab, idx, 4);
                    acc = _mm256_add_ps(acc, val);
                }
            }
            _mm256_storeu_ps(out + oc * N + n, acc);
        }
    }
}

// ── LUT inference: vpshufb nibble decomposition (proper SIMD LUT) ──
// For each weight value w, activation byte a:
//   w * a = w * (hi_nibble * 16 + lo_nibble) = w*hi*16 + w*lo
// Precompute two 16-entry int16 tables per weight:
//   tab_lo[i] = w * i        (i = 0..15)
//   tab_hi[i] = w * i * 16   (i = 0..15)
// vpshufb does 32 lookups from a 16-entry table in one instruction (16 per lane).
// We work in int16 to avoid overflow (max: 255 * 1 = 255, fits int16).

// Build nibble LUT tables: for each weight in [OC][K], store tab_lo and tab_hi as __m128i
// packed into a single __m256i (lo in low 128, hi in high 128).
struct NibbleLUT {
    std::vector<__m256i> tables;  // [OC * K] — one per weight element
    int OC, K;

    void build(const int8_t* weights, int oc_count, int k_count) {
        OC = oc_count; K = k_count;
        tables.resize(OC * K);
        for (int oc = 0; oc < OC; oc++) {
            for (int k = 0; k < K; k++) {
                int8_t w = weights[oc * K + k];
                // tab_lo[i] = w * i as int8 (i=0..15, max |val| = 15, fits int8)
                // tab_hi[i] = w * i * 16 as int16... but we need to fit in bytes for vpshufb
                // Alternative: accumulate in int16 using two vpshufb + unpack
                // Simpler: use int8 tables, accumulate in int16 via maddubs or manual unpack

                // Actually for CNN ternary: w ∈ {-1,0,1}
                // tab_lo[i] = w * i, range [-15, 15] → fits int8
                // tab_hi[i] = w * i * 16, range [-240, 240] → does NOT fit int8
                // So we must work in int16. Use two vpshufb for lo nibble, two for hi nibble.

                // Pack lo table: each entry is w*i as int8 (fits for ternary)
                alignas(16) int8_t lo[16], hi_lo[16], hi_hi[16];
                for (int i = 0; i < 16; i++) {
                    lo[i] = (int8_t)(w * i);          // w*i, range [-15,15]
                    int16_t hv = (int16_t)(w * i * 16); // w*i*16, range [-240,240]
                    hi_lo[i] = (int8_t)(hv & 0xFF);    // low byte
                    hi_hi[i] = (int8_t)(hv >> 8);      // high byte (sign extension)
                }
                // Store: we'll need 3 __m128i per weight but that's too much.
                // Better approach: just store lo and hi_lo, since hi_hi is always 0 or -1 for ternary
                // (range [-240,240], high byte is 0x00 or 0xFF)
                // Actually let's just use a simpler layout.

                // Store lo in low 128 bits, hi_lo in high 128 bits
                __m128i vlo = _mm_loadu_si128((__m128i*)lo);
                __m128i vhi = _mm_loadu_si128((__m128i*)hi_lo);
                tables[oc * K + k] = _mm256_set_m128i(vhi, vlo);
            }
        }
    }
};

// vpshufb-based GEMM: processes 32 activation bytes at a time
// Accumulates in int16, then reduces to int32
void gemm_lut_vpshufb(const uint8_t* activations, // [N][K] row-major
                      const NibbleLUT& lut,
                      int32_t* out,               // [OC][N]
                      int OC, int N, int K) {
    __m256i mask_lo = _mm256_set1_epi8(0x0F);

    for (int oc = 0; oc < OC; oc++) {
        for (int n = 0; n < N; n += 32) {
            __m256i acc_lo = _mm256_setzero_si256();  // int16 accumulators
            __m256i acc_hi = _mm256_setzero_si256();

            for (int k = 0; k < K; k++) {
                // Load 32 activation bytes (32 spatial positions, same k)
                // activations is [N][K], so we need act[n+0..n+31][k]
                // This is strided — we need to gather. For benchmark fairness,
                // let's use a transposed layout [K][N] for the vpshufb path.
                __m256i act = _mm256_loadu_si256((__m256i*)(activations + k * N + n));

                // Split into nibbles
                __m256i lo_nib = _mm256_and_si256(act, mask_lo);
                __m256i hi_nib = _mm256_and_si256(_mm256_srli_epi16(act, 4), mask_lo);

                // Load LUT for this weight
                __m256i tbl = lut.tables[oc * K + k];
                __m128i tbl_lo_128 = _mm256_castsi256_si128(tbl);
                __m128i tbl_hi_128 = _mm256_extracti128_si256(tbl, 1);

                // Broadcast to both lanes
                __m256i tbl_lo = _mm256_broadcastsi128_si256(tbl_lo_128);
                __m256i tbl_hi = _mm256_broadcastsi128_si256(tbl_hi_128);

                // vpshufb: 32 lookups each
                __m256i val_lo = _mm256_shuffle_epi8(tbl_lo, lo_nib);  // w*lo_nibble (int8)
                __m256i val_hi = _mm256_shuffle_epi8(tbl_hi, hi_nib);  // low byte of w*hi_nibble*16

                // Add lo + hi_lo in int8, then widen to int16 and accumulate
                // val_lo range: [-15,15], val_hi range: [-240,240] as low byte
                // We need proper int16 addition. Unpack to int16 first.
                __m256i sum8 = _mm256_add_epi8(val_lo, val_hi); // This can overflow!

                // Safer: sign-extend to int16 and add
                __m256i lo16_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(val_lo));
                __m256i lo16_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(val_lo, 1));
                __m256i hi16_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(val_hi));
                __m256i hi16_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(val_hi, 1));

                acc_lo = _mm256_add_epi16(acc_lo, _mm256_add_epi16(lo16_lo, hi16_lo));
                acc_hi = _mm256_add_epi16(acc_hi, _mm256_add_epi16(lo16_hi, hi16_hi));
            }

            // Reduce int16 accumulators to int32 and store
            // acc_lo has 16 int16 values for positions n+0..n+15
            // acc_hi has 16 int16 values for positions n+16..n+31
            __m256i out0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc_lo));
            __m256i out1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc_lo, 1));
            __m256i out2 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc_hi));
            __m256i out3 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc_hi, 1));
            _mm256_storeu_si256((__m256i*)(out + oc*N + n),      out0);
            _mm256_storeu_si256((__m256i*)(out + oc*N + n + 8),  out1);
            _mm256_storeu_si256((__m256i*)(out + oc*N + n + 16), out2);
            _mm256_storeu_si256((__m256i*)(out + oc*N + n + 24), out3);
        }
    }
}

struct LayerConfig {
    const char* name;
    int OC, N, K;  // N = spatial positions (H*W), K = reduction (IC * kh * kw)
};

int main() {
    // Representative CIFAR-10 ResNet-18 layer sizes
    LayerConfig layers[] = {
        {"layer1 (64OC, 32x32, 3x3)",   64,  1024, 576},   // 64 IC * 3 * 3
        {"layer2 (128OC, 16x16, 3x3)",  128, 256,  1152},   // 128 IC * 3 * 3
        {"layer3 (256OC, 8x8, 3x3)",    256, 64,   2304},   // 256 IC * 3 * 3
        {"layer4 (512OC, 4x4, 3x3)",    512, 16,   4608},   // 512 IC * 3 * 3
    };

    int warmup = 200, runs = 1000;

    printf("╔══════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  dpbusd vs LUT Microbenchmark (1T, %d warmup, %d runs)                         ║\n", warmup, runs);
    printf("╠══════════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║ %-35s │ %8s │ %8s │ %8s │ %8s │ %5s ║\n",
           "Layer", "dpbusd", "LUT-scl", "LUT-gth", "LUT-shuf", "ratio");
    printf("╠══════════════════════════════════════════════════════════════════════════════════╣\n");

    for (auto& L : layers) {
        int OC = L.OC, N = L.N, K = L.K;
        int nkg = K / 4;
        int n_jg = N / 8;

        // Align N to 8 for both
        int N8 = (N + 7) & ~7;

        // Allocate
        std::vector<uint8_t> act_packed(nkg * n_jg * 32);
        std::vector<int32_t> wt_packed(OC * nkg);
        std::vector<int32_t> out_dpbusd(OC * N8);

        std::vector<uint8_t> act_row(N8 * K);
        std::vector<int8_t> weights_i8(OC * K);
        std::vector<float> lut_table(OC * nkg * 256);
        std::vector<float> out_lut_s(OC * N8);
        std::vector<float> out_lut_g(OC * N8);

        // Fill random
        for (auto& v : act_packed) v = rand() % 256;
        for (auto& v : wt_packed) v = rand();
        for (auto& v : act_row) v = rand() % 256;
        for (auto& v : weights_i8) v = (rand() % 3) - 1;  // ternary {-1,0,1}

        build_lut(weights_i8.data(), lut_table.data(), OC, K);

        // Align N to 32 for vpshufb path
        int N32 = (N8 + 31) & ~31;

        // Transposed activations for vpshufb: [K][N] layout
        std::vector<uint8_t> act_kn(K * N32, 0);
        for (int n = 0; n < N8; n++)
            for (int k = 0; k < K; k++)
                act_kn[k * N32 + n] = act_row[n * K + k];

        // Build nibble LUT
        NibbleLUT nlut;
        nlut.build(weights_i8.data(), OC, K);
        std::vector<int32_t> out_shuf(OC * N32);

        // Benchmark dpbusd
        for (int i = 0; i < warmup; i++)
            gemm_dpbusd(act_packed.data(), wt_packed.data(), out_dpbusd.data(), OC, N8, K);
        std::vector<double> t_dpbusd(runs);
        for (int i = 0; i < runs; i++) {
            double t0 = now_ms();
            gemm_dpbusd(act_packed.data(), wt_packed.data(), out_dpbusd.data(), OC, N8, K);
            t_dpbusd[i] = now_ms() - t0;
        }

        // Benchmark LUT scalar
        for (int i = 0; i < warmup; i++)
            gemm_lut_scalar(act_row.data(), lut_table.data(), out_lut_s.data(), OC, N8, K);
        std::vector<double> t_lut_s(runs);
        for (int i = 0; i < runs; i++) {
            double t0 = now_ms();
            gemm_lut_scalar(act_row.data(), lut_table.data(), out_lut_s.data(), OC, N8, K);
            t_lut_s[i] = now_ms() - t0;
        }

        // Benchmark LUT gather
        for (int i = 0; i < warmup; i++)
            gemm_lut_gather(act_row.data(), lut_table.data(), out_lut_g.data(), OC, N8, K);
        std::vector<double> t_lut_g(runs);
        for (int i = 0; i < runs; i++) {
            double t0 = now_ms();
            gemm_lut_gather(act_row.data(), lut_table.data(), out_lut_g.data(), OC, N8, K);
            t_lut_g[i] = now_ms() - t0;
        }

        // Benchmark LUT vpshufb
        for (int i = 0; i < warmup; i++)
            gemm_lut_vpshufb(act_kn.data(), nlut, out_shuf.data(), OC, N32, K);
        std::vector<double> t_lut_sh(runs);
        for (int i = 0; i < runs; i++) {
            double t0 = now_ms();
            gemm_lut_vpshufb(act_kn.data(), nlut, out_shuf.data(), OC, N32, K);
            t_lut_sh[i] = now_ms() - t0;
        }

        std::sort(t_dpbusd.begin(), t_dpbusd.end());
        std::sort(t_lut_s.begin(), t_lut_s.end());
        std::sort(t_lut_g.begin(), t_lut_g.end());
        std::sort(t_lut_sh.begin(), t_lut_sh.end());

        double med_d = t_dpbusd[runs/2];
        double med_ls = t_lut_s[runs/2];
        double med_lg = t_lut_g[runs/2];
        double med_sh = t_lut_sh[runs/2];
        double best_lut = std::min({med_ls, med_lg, med_sh});

        printf("║ %-35s │ %6.3fms │ %6.3fms │ %6.3fms │ %6.3fms │ %4.1fx ║\n",
               L.name, med_d, med_ls, med_lg, med_sh, best_lut / med_d);
    }

    printf("╚══════════════════════════════════════════════════════════════════════════════════╝\n");
    printf("\nratio = best LUT (of all 3) / dpbusd (higher = LUT is slower)\n");
    return 0;
}
