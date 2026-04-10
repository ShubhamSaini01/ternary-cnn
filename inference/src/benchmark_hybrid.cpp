#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <vector>
#include <cmath>
#include <omp.h>

#include "ternary_engine_hybrid.h"

static const float MEAN[3] = {0.4914f, 0.4822f, 0.4465f};
static const float STD[3]  = {0.2023f, 0.1994f, 0.2010f};

Tensor make_random_input(int batch = 1) {
    Tensor input(batch, 3, 32, 32);
    srand(42);
    for (int i = 0; i < input.total(); i++) {
        int c = (i / (32 * 32)) % 3;
        float raw = static_cast<float>(rand()) / RAND_MAX;
        input.data[i] = (raw - MEAN[c]) / STD[c];
    }
    return input;
}

int argmax(const Tensor& logits, int sample = 0) {
    int best = 0;
    float best_val = logits.at(sample, 0, 0, 0);
    for (int i = 1; i < logits.c; i++) {
        float val = logits.at(sample, i, 0, 0);
        if (val > best_val) { best_val = val; best = i; }
    }
    return best;
}

struct BenchResult {
    double mean, median, min, p95;
};

template<typename Engine>
BenchResult run_bench(Engine& engine, const Tensor& input,
                      int warmup, int runs, const char* label) {
    // Warmup
    for (int i = 0; i < warmup; i++)
        engine.forward(input);

    // Benchmark
    std::vector<double> times(runs);
    for (int i = 0; i < runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        engine.forward(input);
        auto end = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    std::sort(times.begin(), times.end());
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    BenchResult r;
    r.mean = sum / runs;
    r.median = times[runs / 2];
    r.min = times.front();
    r.p95 = times[(int)(runs * 0.95)];

    printf("  %-30s  median=%7.3f ms  mean=%7.3f ms  min=%7.3f ms  p95=%7.3f ms\n",
           label, r.median, r.mean, r.min, r.p95);
    return r;
}

int main(int argc, char* argv[]) {
    printf("════════════════════════════════════════════════════════════\n");
    printf("  Hybrid Benchmark: Original VNNI vs oneDNN Hybrid\n");
    printf("════════════════════════════════════════════════════════════\n\n");

    const char* model_path = "export/ternary_resnet18.bin";
    if (argc > 1) model_path = argv[1];
    int warmup = 100, runs = 500;
    if (argc > 2) warmup = atoi(argv[2]);
    if (argc > 3) runs = atoi(argv[3]);

    Tensor input = make_random_input(1);

    // ── Load both engines ──
    printf("Loading Original (V9b-I2S-VNNI) engine...\n");
    TernaryResNet18 orig;
    if (!orig.load(model_path)) { fprintf(stderr, "Load failed\n"); return 1; }

    printf("Loading Hybrid (oneDNN+Custom) engine...\n");
    HybridTernaryResNet18 hybrid;
    if (!hybrid.load(model_path)) { fprintf(stderr, "Load failed\n"); return 1; }

    // ── Sanity check: both engines produce same output ──
    {
        omp_set_num_threads(1);
        Tensor out_orig = orig.forward(input);
        Tensor out_hybrid = hybrid.forward(input);

        printf("\nSanity check — predictions match: ");
        int pred_orig = argmax(out_orig);
        int pred_hybrid = argmax(out_hybrid);
        if (pred_orig == pred_hybrid) {
            printf("YES (class %d)\n", pred_orig);
        } else {
            printf("NO! orig=%d hybrid=%d\n", pred_orig, pred_hybrid);
        }

        // Check max absolute difference
        float max_diff = 0;
        for (int i = 0; i < out_orig.c; i++) {
            float diff = fabsf(out_orig.at(0, i, 0, 0) - out_hybrid.at(0, i, 0, 0));
            if (diff > max_diff) max_diff = diff;
        }
        printf("Max logit difference: %.6f\n", max_diff);
    }

    // ── Thread counts to test ──
    int thread_counts[] = {1, 2, 4, 6};
    int n_configs = sizeof(thread_counts) / sizeof(thread_counts[0]);

    // Check available cores
    int max_threads = omp_get_max_threads();
    printf("\nSystem max threads: %d\n", max_threads);
    printf("Warmup: %d runs, Benchmark: %d runs\n\n", warmup, runs);

    // ── Results storage ──
    struct ConfigResult {
        int threads;
        BenchResult orig, hybrid;
    };
    std::vector<ConfigResult> results;

    for (int ti = 0; ti < n_configs; ti++) {
        int nt = thread_counts[ti];
        if (nt > max_threads) continue;

        printf("────────────────────────────────────────────────────────\n");
        printf("  %d Thread(s)\n", nt);
        printf("────────────────────────────────────────────────────────\n");

        omp_set_num_threads(nt);
        // Also set oneDNN thread count
        // oneDNN uses OMP threads, controlled by omp_set_num_threads

        ConfigResult cr;
        cr.threads = nt;

        orig.reset_profile();
        cr.orig = run_bench(orig, input, warmup, runs, "Original (hand-written VNNI)");

        hybrid.reset_profile();
        cr.hybrid = run_bench(hybrid, input, warmup, runs, "Hybrid (oneDNN conv)");

        double speedup = cr.orig.median / cr.hybrid.median;
        printf("  → Speedup (hybrid/orig): %.2fx %s\n\n",
               speedup > 1.0 ? speedup : 1.0 / speedup,
               speedup > 1.0 ? "FASTER" : "SLOWER");

        results.push_back(cr);
    }

    // ── Print per-layer profile for single-threaded ──
    printf("\n");
    omp_set_num_threads(1);
    // oneDNN uses OMP threads via omp_set_num_threads

    orig.reset_profile();
    for (int i = 0; i < runs; i++) orig.forward(input);
    orig.print_profile();
    printf("(Totals over %d runs)\n", runs);

    hybrid.reset_profile();
    for (int i = 0; i < runs; i++) hybrid.forward(input);
    hybrid.print_profile();
    printf("(Totals over %d runs)\n", runs);

    // ── Summary table ──
    printf("\n════════════════════════════════════════════════════════════\n");
    printf("  SUMMARY (median latency, ms)\n");
    printf("════════════════════════════════════════════════════════════\n");
    printf("  %-10s  %12s  %12s  %10s\n", "Threads", "Original", "Hybrid", "Speedup");
    printf("  ──────────────────────────────────────────────────────\n");
    for (auto& r : results) {
        double sp = r.orig.median / r.hybrid.median;
        printf("  %-10d  %9.3f ms  %9.3f ms  %8.2fx %s\n",
               r.threads, r.orig.median, r.hybrid.median,
               sp > 1.0 ? sp : 1.0 / sp,
               sp > 1.0 ? "faster" : "slower");
    }
    printf("════════════════════════════════════════════════════════════\n");

    return 0;
}
