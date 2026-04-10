#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include "engine_4bit.h"

// ─── CIFAR-10 test data loader ──────────────────────────────

struct TestData {
    std::vector<std::vector<float>> images;
    std::vector<int> labels;
};

static TestData load_cifar10_batch(const char* path) {
    TestData td;
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open: %s\n", path);
        return td;
    }

    uint32_t count;
    if (fread(&count, 4, 1, f) != 1) { fclose(f); return td; }

    for (uint32_t i = 0; i < count; i++) {
        uint32_t label;
        if (fread(&label, 4, 1, f) != 1) break;
        td.labels.push_back(label);

        std::vector<float> img(3 * 32 * 32);
        if (fread(img.data(), 4, 3 * 32 * 32, f) != 3 * 32 * 32) break;
        td.images.push_back(std::move(img));
    }

    fclose(f);
    printf("Loaded %zu test images from %s\n", td.images.size(), path);
    return td;
}

static Tensor make_input(const std::vector<float>& img) {
    Tensor t(1, 3, 32, 32);
    memcpy(t.data.data(), img.data(), 3 * 32 * 32 * sizeof(float));
    return t;
}

static int argmax(const std::vector<float>& v) {
    return std::max_element(v.begin(), v.end()) - v.begin();
}

// ─── Main ───────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.bin> <scales.json> [test_batch_path] [warmup] [runs]\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const char* scales_path = argv[2];
    const char* test_path = argc > 3 ? argv[3] : nullptr;
    int warmup = argc > 4 ? atoi(argv[4]) : 50;
    int runs = argc > 5 ? atoi(argv[5]) : 200;

    // Load model
    Model m = load_model(model_path);
    Engine4bit engine;
    engine.init(m, scales_path);

    // ─── Accuracy test ──────────────────────────────────────
    if (test_path) {
        printf("\n══════════════════════════════════════════════\n");
        printf("ACCURACY TEST (4-bit engine)\n");
        printf("══════════════════════════════════════════════\n");

        TestData td = load_cifar10_batch(test_path);
        if (td.images.empty()) {
            fprintf(stderr, "No test data loaded\n");
            return 1;
        }

        int correct = 0;
        size_t num_test = td.images.size();

        const char* quick_env = getenv("QUICK_TEST");
        if (quick_env) num_test = std::min(num_test, (size_t)atoi(quick_env));

        for (size_t i = 0; i < num_test; i++) {
            Tensor input = make_input(td.images[i]);
            auto output = engine.forward(input);
            int pred = argmax(output);
            if (pred == td.labels[i]) correct++;

            if ((i + 1) % 500 == 0 || i + 1 == num_test) {
                printf("  %zu/%zu  acc=%.2f%%\n", i + 1, num_test,
                       100.0 * correct / (i + 1));
            }
        }

        double acc = 100.0 * correct / num_test;
        printf("\nOverall: %d/%zu = %.2f%%\n", correct, num_test, acc);
    }

    // ─── Latency benchmark ──────────────────────────────────
    printf("\n══════════════════════════════════════════════\n");
    printf("LATENCY BENCHMARK (4-bit engine, sign+add GEMM)\n");
    printf("══════════════════════════════════════════════\n");

    Tensor dummy(1, 3, 32, 32);
    for (auto& v : dummy.data) v = 0.1f;

    printf("Warming up (%d runs)...\n", warmup);
    for (int i = 0; i < warmup; i++) {
        engine.forward(dummy);
    }

    printf("Benchmarking (%d runs)...\n", runs);
    std::vector<double> times(runs);
    for (int i = 0; i < runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        engine.forward(dummy);
        auto end = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    std::sort(times.begin(), times.end());
    double median = times[runs / 2];
    double mean = std::accumulate(times.begin(), times.end(), 0.0) / runs;
    double p95 = times[(int)(runs * 0.95)];
    double min_t = times[0];

    printf("\n  Median: %.3f ms\n", median);
    printf("  Mean:   %.3f ms\n", mean);
    printf("  Min:    %.3f ms\n", min_t);
    printf("  P95:    %.3f ms\n", p95);

    return 0;
}
