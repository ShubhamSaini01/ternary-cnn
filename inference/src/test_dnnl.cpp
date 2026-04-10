#include <cstdio>
#include <chrono>
#include <vector>
#include <cstring>
#include <omp.h>
#include <dnnl.hpp>

int main() {
    using namespace dnnl;

    printf("Creating oneDNN engine...\n");
    engine eng(engine::kind::cpu, 0);
    stream strm(eng);
    printf("Engine created.\n");

    // Test a representative conv: 512x512 3x3 on 4x4 (the largest and most common)
    struct TestCase {
        int ic, oc, h, w, kh, kw, stride, pad;
        const char* name;
    };

    TestCase cases[] = {
        {64,  64,  32, 32, 3, 3, 1, 1, "64→64   32x32"},
        {64,  128, 32, 32, 3, 3, 2, 1, "64→128  32→16"},
        {128, 128, 16, 16, 3, 3, 1, 1, "128→128 16x16"},
        {256, 256,  8,  8, 3, 3, 1, 1, "256→256  8x8 "},
        {512, 512,  4,  4, 3, 3, 1, 1, "512→512  4x4 "},
    };

    omp_set_num_threads(1);
    printf("\nSingle-threaded oneDNN INT8 conv benchmark:\n");
    printf("%-20s  %10s  %10s  %10s\n", "Layer", "Create(ms)", "1st(ms)", "Avg(us)");

    for (auto& tc : cases) {
        int out_h = (tc.h + 2*tc.pad - tc.kh) / tc.stride + 1;
        int out_w = (tc.w + 2*tc.pad - tc.kw) / tc.stride + 1;

        memory::dims src_d = {1, tc.ic, tc.h, tc.w};
        memory::dims wt_d  = {tc.oc, tc.ic, tc.kh, tc.kw};
        memory::dims dst_d = {1, tc.oc, out_h, out_w};
        memory::dims strd  = {tc.stride, tc.stride};
        memory::dims pad_l = {tc.pad, tc.pad};
        memory::dims pad_r = {tc.pad, tc.pad};

        auto src_md = memory::desc(src_d, memory::data_type::u8, memory::format_tag::nchw);
        auto wt_md  = memory::desc(wt_d, memory::data_type::s8, memory::format_tag::any);
        auto dst_md = memory::desc(dst_d, memory::data_type::s32, memory::format_tag::nchw);

        auto t0 = std::chrono::high_resolution_clock::now();

        auto conv_pd = convolution_forward::primitive_desc(
            eng, prop_kind::forward_inference,
            algorithm::convolution_direct,
            src_md, wt_md, dst_md, strd, pad_l, pad_r);

        // Create weight memory and fill with dummy data
        auto user_wt = memory(memory::desc(wt_d, memory::data_type::s8,
                              memory::format_tag::oihw), eng);
        int8_t* wp = (int8_t*)user_wt.get_data_handle();
        int wt_size = tc.oc * tc.ic * tc.kh * tc.kw;
        for (int i = 0; i < wt_size; i++) wp[i] = (i % 3) - 1; // {-1, 0, 1}

        auto wt_mem = memory(conv_pd.weights_desc(), eng);
        reorder(user_wt, wt_mem).execute(strm, user_wt, wt_mem);
        strm.wait();

        auto prim = convolution_forward(conv_pd);

        auto t1 = std::chrono::high_resolution_clock::now();
        double create_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Source and dst buffers
        std::vector<uint8_t> src_data(tc.ic * tc.h * tc.w, 128);
        std::vector<int32_t> dst_data(tc.oc * out_h * out_w, 0);

        // First run (JIT compilation)
        auto t2 = std::chrono::high_resolution_clock::now();
        {
            memory src_mem(src_md, eng, src_data.data());
            memory dst_mem(dst_md, eng, dst_data.data());
            prim.execute(strm, {
                {DNNL_ARG_SRC, src_mem},
                {DNNL_ARG_WEIGHTS, wt_mem},
                {DNNL_ARG_DST, dst_mem}
            });
            strm.wait();
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        double first_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

        // Steady state (100 runs)
        auto t4 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; i++) {
            memory src_mem(src_md, eng, src_data.data());
            memory dst_mem(dst_md, eng, dst_data.data());
            prim.execute(strm, {
                {DNNL_ARG_SRC, src_mem},
                {DNNL_ARG_WEIGHTS, wt_mem},
                {DNNL_ARG_DST, dst_mem}
            });
            strm.wait();
        }
        auto t5 = std::chrono::high_resolution_clock::now();
        double avg_us = std::chrono::duration<double, std::micro>(t5 - t4).count() / 100.0;

        printf("%-20s  %10.1f  %10.1f  %10.1f\n", tc.name, create_ms, first_ms, avg_us);
    }

    printf("\nDone.\n");
    return 0;
}
