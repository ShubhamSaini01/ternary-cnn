#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// ─── Accuracy verification on CIFAR-10 test set ─────────────
void run_accuracy(TernaryResNet18& engine, const char* test_path) {
    FILE* f = fopen(test_path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open test data: %s\n", test_path);
        fprintf(stderr, "Run: python export/export_test_data.py\n");
        return;
    }

    uint32_t num_images;
    fread(&num_images, 4, 1, f);
    printf("Running accuracy on %u images...\n\n", num_images);

    const char* class_names[10] = {
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    };
    int class_correct[10] = {0};
    int class_total[10] = {0};
    int correct = 0, total = 0;

    std::vector<float> pixels(3 * 32 * 32);
    auto start = std::chrono::high_resolution_clock::now();

    for (uint32_t i = 0; i < num_images; i++) {
        fread(pixels.data(), 4, 3 * 32 * 32, f);
        uint32_t label;
        fread(&label, 4, 1, f);

        Tensor input(1, 3, 32, 32);
        memcpy(input.data.data(), pixels.data(), 3 * 32 * 32 * sizeof(float));

        Tensor logits = engine.forward(input);
        int pred = argmax(logits);

        class_total[label]++;
        if (pred == (int)label) { correct++; class_correct[label]++; }
        total++;

        if ((i + 1) % 1000 == 0 || i + 1 == num_images) {
            printf("  [%5u/%u] accuracy: %.2f%%\n", i + 1, num_images,
                   100.0 * correct / total);
        }
    }
    fclose(f);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    printf("\n────────────────────────────────────────\n");
    printf("ACCURACY RESULTS\n");
    printf("────────────────────────────────────────\n");
    printf("  Overall: %d / %d = %.2f%%\n", correct, total, 100.0 * correct / total);
    printf("  Time: %.1f sec (%.1f ms/image)\n\n", elapsed, 1000 * elapsed / total);

    printf("  %-12s  %6s  %6s  %8s\n", "Class", "Right", "Total", "Accuracy");
    printf("  ────────────────────────────────────\n");
    for (int c = 0; c < 10; c++) {
        float acc = class_total[c] > 0 ? 100.0f * class_correct[c] / class_total[c] : 0;
        printf("  %-12s  %6d  %6d  %7.2f%%\n",
               class_names[c], class_correct[c], class_total[c], acc);
    }
    printf("────────────────────────────────────────\n");
}

// ─── Benchmark mode ──────────────────────────────────────────
void run_benchmark(TernaryResNet18& engine, int warmup_runs, int benchmark_runs) {
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

    engine.reset_profile();

    for (int i = 0; i < benchmark_runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        Tensor logits = engine.forward(input);
        auto end = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(end - start).count();
    }

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

    engine.print_profile();
    printf("(Profile times are totals over %d runs. Divide by %d for per-inference.)\n",
           benchmark_runs, benchmark_runs);

    printf("\n============================================================\n");
    printf("SUMMARY\n");
    printf("============================================================\n");
    printf("  Latency (median): %.3f ms\n", median);
    printf("  Throughput:       %.1f img/s\n", 1000.0 / median);
    printf("============================================================\n");
}

int main(int argc, char* argv[]) {
    printf("============================================================\n");
    printf("Ternary CNN — C++ SIMD Inference Engine\n");
    printf("============================================================\n\n");

    printf("Usage:\n");
    printf("  Benchmark: ./ternary_inference <model.bin> <warmup> <runs>\n");
    printf("  Accuracy:  ./ternary_inference <model.bin> --accuracy <test.bin>\n\n");

    const char* model_path = "exports/ternary_resnet18.bin";
    if (argc > 1) model_path = argv[1];

    TernaryResNet18 engine;
    if (!engine.load(model_path)) {
        fprintf(stderr, "Failed to load model.\n");
        return 1;
    }

    // Check for accuracy mode
    bool accuracy_mode = false;
    const char* test_path = "exports/cifar10_test.bin";
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--accuracy") == 0) {
            accuracy_mode = true;
            if (i + 1 < argc) test_path = argv[i + 1];
            break;
        }
    }

    if (accuracy_mode) {
        run_accuracy(engine, test_path);
    } else {
        int warmup = 50, runs = 200;
        if (argc > 2) warmup = atoi(argv[2]);
        if (argc > 3) runs = atoi(argv[3]);
        run_benchmark(engine, warmup, runs);
    }

    return 0;
}