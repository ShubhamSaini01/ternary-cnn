#include <cstdio>
#include <chrono>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <omp.h>
#include <dnnl.hpp>

int main() {
    using namespace dnnl;

    engine eng(engine::kind::cpu, 0);
    stream strm(eng);

    // After im2col, we have:
    //   Weights: [OC, K] s8, where K = IC*KH*KW
    //   Activations: [K, N] u8, where N = OH*OW
    //   Output: [OC, N] s32
    // This is a GEMM: C = A × B where A=[OC,K], B=[K,N], C=[OC,N]

    struct TestCase {
        int oc, k, n;  // OC, IC*KH*KW, OH*OW
        const char* name;
    };

    TestCase cases[] = {
        {64,   576,  1024, "64→64   32x32 (K=576, N=1024)"},
        {128,  576,   256, "64→128  s=2   (K=576, N=256) "},
        {128, 1152,   256, "128→128 16x16 (K=1152, N=256)"},
        {256, 1152,    64, "128→256 s=2   (K=1152, N=64) "},
        {256, 2304,    64, "256→256  8x8  (K=2304, N=64) "},
        {512, 2304,    16, "256→512 s=2   (K=2304, N=16) "},
        {512, 4608,    16, "512→512  4x4  (K=4608, N=16) "},
    };

    omp_set_num_threads(1);
    printf("oneDNN INT8 matmul benchmark (1 thread):\n");
    printf("%-40s  %10s  %10s  %10s\n", "Config", "Create(ms)", "1st(us)", "Avg(us)");

    for (auto& tc : cases) {
        memory::dims a_dims = {tc.oc, tc.k};   // weights [OC, K]
        memory::dims b_dims = {tc.k, tc.n};    // activations [K, N]
        memory::dims c_dims = {tc.oc, tc.n};   // output [OC, N]

        auto a_md = memory::desc(a_dims, memory::data_type::s8, memory::format_tag::ab);
        auto b_md = memory::desc(b_dims, memory::data_type::u8, memory::format_tag::ab);
        auto c_md = memory::desc(c_dims, memory::data_type::s32, memory::format_tag::ab);

        auto t0 = std::chrono::high_resolution_clock::now();

        auto matmul_pd = matmul::primitive_desc(eng, a_md, b_md, c_md);
        auto matmul_prim = matmul(matmul_pd);

        // Create weight memory
        auto a_mem = memory(a_md, eng);
        int8_t* ap = (int8_t*)a_mem.get_data_handle();
        for (int i = 0; i < tc.oc * tc.k; i++) ap[i] = (i % 3) - 1;

        auto t1 = std::chrono::high_resolution_clock::now();
        double create_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Activation and output buffers
        std::vector<uint8_t> b_data(tc.k * tc.n, 128);
        std::vector<int32_t> c_data(tc.oc * tc.n, 0);

        // First run
        auto t2 = std::chrono::high_resolution_clock::now();
        {
            memory b_mem(b_md, eng, b_data.data());
            memory c_mem(c_md, eng, c_data.data());
            matmul_prim.execute(strm, {
                {DNNL_ARG_SRC, a_mem},
                {DNNL_ARG_WEIGHTS, b_mem},
                {DNNL_ARG_DST, c_mem}
            });
            strm.wait();
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        double first_us = std::chrono::duration<double, std::micro>(t3 - t2).count();

        // Warmup
        for (int i = 0; i < 50; i++) {
            memory b_mem(b_md, eng, b_data.data());
            memory c_mem(c_md, eng, c_data.data());
            matmul_prim.execute(strm, {
                {DNNL_ARG_SRC, a_mem},
                {DNNL_ARG_WEIGHTS, b_mem},
                {DNNL_ARG_DST, c_mem}
            });
            strm.wait();
        }

        // Steady state (500 runs)
        auto t4 = std::chrono::high_resolution_clock::now();
        int nruns = 500;
        for (int i = 0; i < nruns; i++) {
            memory b_mem(b_md, eng, b_data.data());
            memory c_mem(c_md, eng, c_data.data());
            matmul_prim.execute(strm, {
                {DNNL_ARG_SRC, a_mem},
                {DNNL_ARG_WEIGHTS, b_mem},
                {DNNL_ARG_DST, c_mem}
            });
            strm.wait();
        }
        auto t5 = std::chrono::high_resolution_clock::now();
        double avg_us = std::chrono::duration<double, std::micro>(t5 - t4).count() / nruns;

        printf("%-40s  %10.1f  %10.1f  %10.1f\n", tc.name, create_ms, first_us, avg_us);
    }

    printf("\nDone.\n");
    return 0;
}
