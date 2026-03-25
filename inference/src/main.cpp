#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <vector>
#include <cmath>
#include "ternary_engine.h"

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
    int num_classes = logits.c;
    int best = 0;
    float best_val = logits.at(sample, 0, 0, 0);
    for (int i = 1; i < num_classes; i++) {
        float val = logits.at(sample, i, 0, 0);
        if (val > best_val) { best_val = val; best = i; }
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
    printf("Ternary CNN — C++ SIMD Inference Benchmark (Profiled)\n");
    printf("============================================================\n\n");

    TernaryResNet18 engine;
    if (!engine.load(model_path)) return 1;

    Tensor input = make_random_input(1);
    printf("Input shape: [1, 3, 32, 32]\n\n");

    // Sanity check
    {
        Tensor logits = engine.forward(input);
        printf("Sanity check — logits: [");
        for (int i = 0; i < 10; i++)
            printf("%.4f%s", logits.at(0, i, 0, 0), i < 9 ? ", " : "");
        printf("]\nPredicted class: %d\n\n", argmax(logits));
    }

    // Warmup
    printf("Warming up (%d runs)...\n", warmup_runs);
    for (int i = 0; i < warmup_runs; i++)
        engine.forward(input);

    // Benchmark
    printf("Benchmarking single image (%d runs)...\n", benchmark_runs);
    std::vector<double> times(benchmark_runs);

    // Reset profiler and accumulate over all benchmark runs
    engine.reset_profile();

    for (int i = 0; i < benchmark_runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        Tensor logits = engine.forward(input);
        auto end = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    // Latency stats
    std::sort(times.begin(), times.end());
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / benchmark_runs;
    double median = times[benchmark_runs / 2];

    double var = 0;
    for (auto t : times) var += (t - mean) * (t - mean);
    double std_dev = sqrt(var / benchmark_runs);

    printf("\n────────────────────────────────────────\n");
    printf("Single Image Latency (batch_size=1)\n");
    printf("────────────────────────────────────────\n");
    printf("  Mean:   %.3f ms\n", mean);
    printf("  Median: %.3f ms\n", median);
    printf("  Std:    %.3f ms\n", std_dev);
    printf("  Min:    %.3f ms\n", times.front());
    printf("  Max:    %.3f ms\n", times.back());
    printf("  P95:    %.3f ms\n", times[(int)(benchmark_runs * 0.95)]);
    printf("  P99:    %.3f ms\n", times[(int)(benchmark_runs * 0.99)]);

    // Print profile (averaged over benchmark_runs)
    engine.print_profile();
    printf("(Profile times are totals over %d runs. Divide by %d for per-inference.)\n",
           benchmark_runs, benchmark_runs);

    printf("\n============================================================\n");
    printf("SUMMARY\n");
    printf("============================================================\n");
    printf("  Latency (median): %.3f ms\n", median);
    printf("  Throughput:       %.1f img/s\n", 1000.0 / median);
    printf("============================================================\n");

    return 0;
}