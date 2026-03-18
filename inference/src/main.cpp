#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <vector>
#include <cmath>
#include "ternary_engine.h"

// CIFAR-10 normalization constants
static const float MEAN[3] = {0.4914f, 0.4822f, 0.4465f};
static const float STD[3]  = {0.2023f, 0.1994f, 0.2010f};

// Create a random normalized input (simulates a CIFAR-10 image)
Tensor make_random_input(int batch = 1) {
    Tensor input(batch, 3, 32, 32);
    for (int i = 0; i < input.total(); i++) {
        // Random float in [0, 1], then normalize
        int c = (i / (32 * 32)) % 3;
        float raw = static_cast<float>(rand()) / RAND_MAX;
        input.data[i] = (raw - MEAN[c]) / STD[c];
    }
    return input;
}

int argmax(const Tensor& logits, int sample = 0) {
    int num_classes = logits.c;
    int best = 0;
    float best_val = logits.at(sample, 0, 0, 0);
    for (int i = 1; i < num_classes; i++) {
        float val = logits.at(sample, i, 0, 0);
        if (val > best_val) {
            best_val = val;
            best = i;
        }
    }
    return best;
}

int main(int argc, char* argv[]) {
    const char* model_path = "export/ternary_resnet18.bin";
    int warmup_runs = 50;
    int benchmark_runs = 200;

    if (argc > 1) model_path = argv[1];
    if (argc > 2) warmup_runs = atoi(argv[2]);
    if (argc > 3) benchmark_runs = atoi(argv[3]);

    printf("============================================================\n");
    printf("Ternary CNN — C++ SIMD Inference Benchmark\n");
    printf("============================================================\n\n");

    // Load model
    TernaryResNet18 engine;
    if (!engine.load(model_path)) {
        fprintf(stderr, "Failed to load model.\n");
        return 1;
    }

    // Create input
    Tensor input = make_random_input(1);
    printf("Input shape: [1, 3, 32, 32]\n\n");

    // Sanity check: run once and show output
    {
        Tensor logits = engine.forward(input);
        printf("Sanity check — logits: [");
        for (int i = 0; i < 10; i++) {
            printf("%.4f%s", logits.at(0, i, 0, 0), i < 9 ? ", " : "");
        }
        printf("]\n");
        printf("Predicted class: %d\n\n", argmax(logits));
    }

    // ─── Warmup ──────────────────────────────────────────────
    printf("Warming up (%d runs)...\n", warmup_runs);
    for (int i = 0; i < warmup_runs; i++) {
        Tensor logits = engine.forward(input);
    }

    // ─── Benchmark: Single image latency ─────────────────────
    printf("Benchmarking single image (%d runs)...\n", benchmark_runs);
    std::vector<double> times(benchmark_runs);

    for (int i = 0; i < benchmark_runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        Tensor logits = engine.forward(input);
        auto end = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    // Stats
    std::sort(times.begin(), times.end());
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / benchmark_runs;
    double median = times[benchmark_runs / 2];
    double min_t = times.front();
    double max_t = times.back();
    double p95 = times[(int)(benchmark_runs * 0.95)];
    double p99 = times[(int)(benchmark_runs * 0.99)];

    double var = 0;
    for (auto t : times) var += (t - mean) * (t - mean);
    double std_dev = sqrt(var / benchmark_runs);

    printf("\n");
    printf("────────────────────────────────────────\n");
    printf("Single Image Latency (batch_size=1)\n");
    printf("────────────────────────────────────────\n");
    printf("  Mean:   %.3f ms\n", mean);
    printf("  Median: %.3f ms\n", median);
    printf("  Std:    %.3f ms\n", std_dev);
    printf("  Min:    %.3f ms\n", min_t);
    printf("  Max:    %.3f ms\n", max_t);
    printf("  P95:    %.3f ms\n", p95);
    printf("  P99:    %.3f ms\n", p99);

    // ─── Summary ─────────────────────────────────────────────
    printf("\n");
    printf("============================================================\n");
    printf("SUMMARY — C++ SIMD Ternary Inference\n");
    printf("============================================================\n");
    printf("  Latency (median): %.3f ms\n", median);
    printf("  Throughput:       %.1f img/s\n", 1000.0 / median);
    printf("============================================================\n");

    return 0;
}
