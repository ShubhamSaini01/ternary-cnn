#include <cstdio>
#include <chrono>
#include <vector>
#include <cstring>
#include <omp.h>
#include <dnnl.hpp>

int main() {
    using namespace dnnl;

    engine eng(engine::kind::cpu, 0);
    stream strm(eng);

    struct TestCase {
        int oc, k, n;
        const char* name;
    };

    TestCase cases[] = {
        {64,   576,  1024, "64→64   32x32"},
        {128, 1152,   256, "128→128 16x16"},
        {256, 2304,    64, "256→256  8x8 "},
        {512, 4608,    16, "512→512  4x4 "},
    };

    omp_set_num_threads(1);
    printf("oneDNN INT8 matmul (pre-alloc memory, 1 thread):\n");
    printf("%-20s  %10s  %10s\n", "Config", "1st(us)", "Avg(us)");

    for (auto& tc : cases) {
        auto a_md = memory::desc({tc.oc, tc.k}, memory::data_type::s8, memory::format_tag::ab);
        auto b_md = memory::desc({tc.k, tc.n}, memory::data_type::u8, memory::format_tag::ab);
        auto c_md = memory::desc({tc.oc, tc.n}, memory::data_type::s32, memory::format_tag::ab);

        auto matmul_pd = matmul::primitive_desc(eng, a_md, b_md, c_md);
        auto matmul_prim = matmul(matmul_pd);

        // Pre-allocate ALL memory objects
        auto a_mem = memory(a_md, eng);
        auto b_mem = memory(b_md, eng);
        auto c_mem = memory(c_md, eng);

        // Fill weights
        int8_t* ap = (int8_t*)a_mem.get_data_handle();
        for (int i = 0; i < tc.oc * tc.k; i++) ap[i] = (i % 3) - 1;

        // Fill activations
        uint8_t* bp = (uint8_t*)b_mem.get_data_handle();
        memset(bp, 128, tc.k * tc.n);

        // First run
        auto t0 = std::chrono::high_resolution_clock::now();
        matmul_prim.execute(strm, {
            {DNNL_ARG_SRC, a_mem},
            {DNNL_ARG_WEIGHTS, b_mem},
            {DNNL_ARG_DST, c_mem}
        });
        strm.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        double first_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

        // Warmup
        for (int i = 0; i < 100; i++) {
            matmul_prim.execute(strm, {
                {DNNL_ARG_SRC, a_mem},
                {DNNL_ARG_WEIGHTS, b_mem},
                {DNNL_ARG_DST, c_mem}
            });
            strm.wait();
        }

        // Benchmark
        int nruns = 1000;
        auto t2 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < nruns; i++) {
            matmul_prim.execute(strm, {
                {DNNL_ARG_SRC, a_mem},
                {DNNL_ARG_WEIGHTS, b_mem},
                {DNNL_ARG_DST, c_mem}
            });
            strm.wait();
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        double avg_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / nruns;

        printf("%-20s  %10.1f  %10.1f\n", tc.name, first_us, avg_us);
    }

    // Also test plain FP32 matmul for comparison
    printf("\noneDNN FP32 matmul (pre-alloc, 1 thread):\n");
    printf("%-20s  %10s  %10s\n", "Config", "1st(us)", "Avg(us)");

    for (auto& tc : cases) {
        auto a_md = memory::desc({tc.oc, tc.k}, memory::data_type::f32, memory::format_tag::ab);
        auto b_md = memory::desc({tc.k, tc.n}, memory::data_type::f32, memory::format_tag::ab);
        auto c_md = memory::desc({tc.oc, tc.n}, memory::data_type::f32, memory::format_tag::ab);

        auto matmul_pd = matmul::primitive_desc(eng, a_md, b_md, c_md);
        auto matmul_prim = matmul(matmul_pd);

        auto a_mem = memory(a_md, eng);
        auto b_mem = memory(b_md, eng);
        auto c_mem = memory(c_md, eng);

        // Warmup
        for (int i = 0; i < 100; i++) {
            matmul_prim.execute(strm, {
                {DNNL_ARG_SRC, a_mem},
                {DNNL_ARG_WEIGHTS, b_mem},
                {DNNL_ARG_DST, c_mem}
            });
            strm.wait();
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        matmul_prim.execute(strm, {
            {DNNL_ARG_SRC, a_mem},
            {DNNL_ARG_WEIGHTS, b_mem},
            {DNNL_ARG_DST, c_mem}
        });
        strm.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        double first_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

        int nruns = 1000;
        auto t2 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < nruns; i++) {
            matmul_prim.execute(strm, {
                {DNNL_ARG_SRC, a_mem},
                {DNNL_ARG_WEIGHTS, b_mem},
                {DNNL_ARG_DST, c_mem}
            });
            strm.wait();
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        double avg_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / nruns;

        printf("%-20s  %10.1f  %10.1f\n", tc.name, first_us, avg_us);
    }

    printf("\nDone.\n");
    return 0;
}
