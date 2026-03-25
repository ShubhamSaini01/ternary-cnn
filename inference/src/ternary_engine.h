#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <immintrin.h>
#include "model_loader.h"

// ═══════════════════════════════════════════════════════════════
// V9: True INT16 SIMD accumulation (BitNet I2_S for CNN)
//
// perf stat proved: IPC=3.81, L1 miss=4% → compute-bound
// FP32 gives 8 adds/instruction, INT16 gives 16 → 2x throughput
//
// 1. Im2col → global INT8 quantize (per-tensor absmax)
// 2. Inner loop: 16 INT16 adds per _mm256_add_epi16
// 3. Flush INT16→INT32 every 200 rows (overflow safety)
// 4. Dequant once per OC: float = int32 * inv_scale * fused_scale + bias
// ═══════════════════════════════════════════════════════════════

#define PROFILE_ENGINE 1

struct EngineProfile {
    double t_im2col=0, t_quantize=0, t_compute=0, t_dequant=0;
    int ternary_calls=0;
    double t_fp32_im2col=0, t_fp32_gemm=0, t_fp32_bn=0;
    int fp32_calls=0;
    double t_relu=0, t_add=0, t_pool=0, t_fc=0;
    void print() {
        double t_tern = t_im2col+t_quantize+t_compute+t_dequant;
        double t_fp = t_fp32_im2col+t_fp32_gemm+t_fp32_bn;
        double total = t_tern+t_fp+t_relu+t_add+t_pool+t_fc;
        printf("\n═══════════════════════════════════════════════════\n");
        printf("ENGINE PROFILE (V9 INT16 accum)\n");
        printf("═══════════════════════════════════════════════════\n\n");
        printf("Ternary Conv (%d calls):           %8.3f ms\n", ternary_calls, t_tern);
        printf("  ├─ im2col:                       %8.3f ms  (%4.1f%%)\n", t_im2col, 100*t_im2col/total);
        printf("  ├─ quantize to INT8:             %8.3f ms  (%4.1f%%)\n", t_quantize, 100*t_quantize/total);
        printf("  ├─ INT16 add/sub compute:        %8.3f ms  (%4.1f%%)\n", t_compute, 100*t_compute/total);
        printf("  └─ dequant + scale+bias:         %8.3f ms  (%4.1f%%)\n", t_dequant, 100*t_dequant/total);
        printf("\nFP32 Conv (%d calls):              %8.3f ms\n", fp32_calls, t_fp);
        printf("  ├─ im2col:                       %8.3f ms  (%4.1f%%)\n", t_fp32_im2col, 100*t_fp32_im2col/total);
        printf("  ├─ GEMM:                         %8.3f ms  (%4.1f%%)\n", t_fp32_gemm, 100*t_fp32_gemm/total);
        printf("  └─ BN:                           %8.3f ms  (%4.1f%%)\n", t_fp32_bn, 100*t_fp32_bn/total);
        printf("\nOther:\n");
        printf("  ├─ ReLU:                         %8.3f ms  (%4.1f%%)\n", t_relu, 100*t_relu/total);
        printf("  ├─ Add:                          %8.3f ms  (%4.1f%%)\n", t_add, 100*t_add/total);
        printf("  ├─ Pool:                         %8.3f ms  (%4.1f%%)\n", t_pool, 100*t_pool/total);
        printf("  └─ FC:                           %8.3f ms  (%4.1f%%)\n", t_fc, 100*t_fc/total);
        printf("\n───────────────────────────────────────────────────\n");
        printf("TOTAL:                             %8.3f ms\n", total);
        printf("═══════════════════════════════════════════════════\n");
    }
    void reset() { *this = EngineProfile{}; }
};
static EngineProfile g_prof;
struct PTimer {
    std::chrono::high_resolution_clock::time_point s;
    void start() { s = std::chrono::high_resolution_clock::now(); }
    double stop_ms() { return std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-s).count(); }
};

struct Tensor {
    std::vector<float> data;
    int n,c,h,w;
    Tensor():n(0),c(0),h(0),w(0){}
    Tensor(int n,int c,int h,int w):data(n*c*h*w,0.0f),n(n),c(c),h(h),w(w){}
    float& at(int ni,int ci,int hi,int wi){return data[((ni*c+ci)*h+hi)*w+wi];}
    const float& at(int ni,int ci,int hi,int wi)const{return data[((ni*c+ci)*h+hi)*w+wi];}
    float* channel_ptr(int ni,int ci){return data.data()+((ni*c+ci)*h)*w;}
    const float* channel_ptr(int ni,int ci)const{return data.data()+((ni*c+ci)*h)*w;}
    int total()const{return n*c*h*w;}
};

static void im2col_rowmajor(const Tensor& input,int ni,int kH,int kW,int stride,int padding,int out_h,int out_w,float* col){
    int in_c=input.c,in_h=input.h,in_w=input.w,cc=out_h*out_w;
    int idx=0;
    for(int ic=0;ic<in_c;ic++){
        const float* ch=input.channel_ptr(ni,ic);
        for(int kh=0;kh<kH;kh++)for(int kw=0;kw<kW;kw++)
            for(int oh=0;oh<out_h;oh++){int ih=oh*stride-padding+kh;
                for(int ow=0;ow<out_w;ow++){int iw=ow*stride-padding+kw;
                    col[idx++]=(ih>=0&&ih<in_h&&iw>=0&&iw<in_w)?ch[ih*in_w+iw]:0.0f;}}
    }
}

static Tensor conv2d_fp32(const Tensor& input,const FP32ConvLayer& layer){
    PTimer pt; auto& p=layer.params;
    int oh=(input.h+2*p.padding-p.kernel_h)/p.stride+1;
    int ow=(input.w+2*p.padding-p.kernel_w)/p.stride+1;
    Tensor output(input.n,p.out_channels,oh,ow);
    int cr=p.in_channels*p.kernel_h*p.kernel_w,cc=oh*ow;
    std::vector<float> col(cr*cc);
    for(int ni=0;ni<input.n;ni++){
        pt.start();im2col_rowmajor(input,ni,p.kernel_h,p.kernel_w,p.stride,p.padding,oh,ow,col.data());
#if PROFILE_ENGINE
        g_prof.t_fp32_im2col+=pt.stop_ms();
#endif
        pt.start();
        for(uint32_t oc=0;oc<p.out_channels;oc++){
            const float* wr=layer.weights.data()+oc*cr;
            float* out=output.channel_ptr(ni,oc);
            memset(out,0,cc*sizeof(float));
            for(int k=0;k<cr;k++){const float* cr2=col.data()+k*cc;float wk=wr[k];int j=0;
#ifdef __AVX2__
                __m256 wv=_mm256_set1_ps(wk);
                for(;j+7<cc;j+=8){__m256 a=_mm256_loadu_ps(out+j);a=_mm256_fmadd_ps(wv,_mm256_loadu_ps(cr2+j),a);_mm256_storeu_ps(out+j,a);}
#endif
                for(;j<cc;j++)out[j]+=wk*cr2[j];}
        }
#if PROFILE_ENGINE
        g_prof.t_fp32_gemm+=pt.stop_ms();
#endif
        pt.start();
        for(uint32_t oc=0;oc<p.out_channels;oc++){
            float* out=output.channel_ptr(ni,oc);
            float s=layer.fused_scale.empty()?1.0f:layer.fused_scale[oc];
            float b=layer.fused_bias.empty()?0.0f:layer.fused_bias[oc];
            int j=0;
#ifdef __AVX2__
            __m256 sv=_mm256_set1_ps(s),bv=_mm256_set1_ps(b);
            for(;j+7<cc;j+=8){__m256 v=_mm256_loadu_ps(out+j);_mm256_storeu_ps(out+j,_mm256_fmadd_ps(v,sv,bv));}
#endif
            for(;j<cc;j++)out[j]=s*out[j]+b;}
#if PROFILE_ENGINE
        g_prof.t_fp32_bn+=pt.stop_ms();g_prof.fp32_calls++;
#endif
    }
    return output;
}

struct TernaryKernel {
    std::vector<std::vector<int>> pos_indices,neg_indices;
    uint32_t out_channels; int col_rows;
    void prepare(const TernaryConvLayer& layer){
        auto& p=layer.params; out_channels=p.out_channels;
        col_rows=p.in_channels*p.kernel_h*p.kernel_w;
        pos_indices.resize(out_channels); neg_indices.resize(out_channels);
        for(uint32_t oc=0;oc<out_channels;oc++){
            pos_indices[oc].clear();neg_indices[oc].clear();
            int bo=oc*col_rows;
            for(int k=0;k<col_rows;k++){int bi=bo+k;
                if((layer.mask_pos[bi>>3]>>(bi&7))&1)pos_indices[oc].push_back(k);
                else if((layer.mask_neg[bi>>3]>>(bi&7))&1)neg_indices[oc].push_back(k);}}
    }
};

// ─── V9 Ternary Conv: INT16 accumulation ─────────────────────
static Tensor conv2d_ternary_int16(const Tensor& input, const TernaryConvLayer& layer,
                                    TernaryKernel& kernel) {
    PTimer pt;
    auto& p = layer.params;
    int out_h = (input.h + 2*p.padding - p.kernel_h) / p.stride + 1;
    int out_w = (input.w + 2*p.padding - p.kernel_w) / p.stride + 1;
    Tensor output(input.n, p.out_channels, out_h, out_w);
    int col_rows = kernel.col_rows;
    int col_cols = out_h * out_w;
    int cc16 = ((col_cols + 15) / 16) * 16; // pad to 16

    std::vector<float> col_data(col_rows * col_cols);
    std::vector<int8_t> col_i8(col_rows * cc16, 0);
    std::vector<int32_t> acc32(cc16);

    const int FLUSH = 200; // flush INT16→INT32 every 200 rows

    for (int ni = 0; ni < input.n; ni++) {
        // ── Im2col ───────────────────────────────
        pt.start();
        im2col_rowmajor(input, ni, p.kernel_h, p.kernel_w, p.stride, p.padding,
                        out_h, out_w, col_data.data());
#if PROFILE_ENGINE
        g_prof.t_im2col += pt.stop_ms();
#endif

        // ── Global INT8 quantize ─────────────────
        pt.start();
        float gmax = 0.0f;
        {
            int tot = col_rows * col_cols, i = 0;
#ifdef __AVX2__
            __m256 vm = _mm256_setzero_ps(), sm = _mm256_set1_ps(-0.0f);
            for (; i+7 < tot; i+=8) {
                __m256 v = _mm256_andnot_ps(sm, _mm256_loadu_ps(col_data.data()+i));
                vm = _mm256_max_ps(vm, v);
            }
            __m128 h=_mm256_extractf128_ps(vm,1),l=_mm256_castps256_ps128(vm);
            __m128 m=_mm_max_ps(l,h);
            m=_mm_max_ps(m,_mm_shuffle_ps(m,m,_MM_SHUFFLE(2,3,0,1)));
            m=_mm_max_ps(m,_mm_shuffle_ps(m,m,_MM_SHUFFLE(1,0,3,2)));
            gmax=_mm_cvtss_f32(m);
#endif
            for (; i < tot; i++) { float v=fabsf(col_data[i]); if(v>gmax)gmax=v; }
        }
        float inv_scale = (gmax > 1e-10f) ? gmax / 127.0f : 0.0f;
        float qscale = (gmax > 1e-10f) ? 127.0f / gmax : 0.0f;

        // Quantize row by row into padded INT8 buffer
        for (int k = 0; k < col_rows; k++) {
            const float* src = col_data.data() + k * col_cols;
            int8_t* dst = col_i8.data() + k * cc16;
            int j = 0;
#ifdef __AVX2__
            __m256 qv = _mm256_set1_ps(qscale);
            for (; j+7 < col_cols; j+=8) {
                __m256 v = _mm256_mul_ps(_mm256_loadu_ps(src+j), qv);
                __m256i i32 = _mm256_cvtps_epi32(v);
                __m256i p16 = _mm256_packs_epi32(i32, i32);
                p16 = _mm256_permute4x64_epi64(p16, 0xD8);
                __m128i i8 = _mm_packs_epi16(_mm256_castsi256_si128(p16),
                                              _mm256_castsi256_si128(p16));
                _mm_storel_epi64((__m128i*)(dst+j), i8);
            }
#endif
            for (; j < col_cols; j++) {
                int v = (int)roundf(src[j] * qscale);
                dst[j] = (int8_t)(v>127?127:(v<-128?-128:v));
            }
            for (int jj = col_cols; jj < cc16; jj++) dst[jj] = 0;
        }
#if PROFILE_ENGINE
        g_prof.t_quantize += pt.stop_ms();
#endif

        // ── INT16 accumulate per output channel ──
        pt.start();
        for (uint32_t oc = 0; oc < p.out_channels; oc++) {
            const auto& pos = kernel.pos_indices[oc];
            const auto& neg = kernel.neg_indices[oc];

            memset(acc32.data(), 0, cc16 * sizeof(int32_t));

            // +1 weights: accumulate in INT16 chunks, flush to INT32
            int pi = 0;
            while (pi < (int)pos.size()) {
                int end = std::min(pi + FLUSH, (int)pos.size());
                int j = 0;
#ifdef __AVX2__
                for (; j+15 < cc16; j+=16) {
                    __m256i s16 = _mm256_setzero_si256();
                    for (int ci = pi; ci < end; ci++) {
                        __m128i b = _mm_loadu_si128((__m128i*)(col_i8.data() + pos[ci]*cc16 + j));
                        s16 = _mm256_add_epi16(s16, _mm256_cvtepi8_epi16(b));
                    }
                    // Widen INT16 → INT32
                    __m128i lo16 = _mm256_castsi256_si128(s16);
                    __m128i hi16 = _mm256_extracti128_si256(s16, 1);
                    __m256i lo32 = _mm256_cvtepi16_epi32(lo16);
                    __m256i hi32 = _mm256_cvtepi16_epi32(hi16);
                    __m256i a0 = _mm256_loadu_si256((__m256i*)(acc32.data()+j));
                    __m256i a1 = _mm256_loadu_si256((__m256i*)(acc32.data()+j+8));
                    _mm256_storeu_si256((__m256i*)(acc32.data()+j), _mm256_add_epi32(a0,lo32));
                    _mm256_storeu_si256((__m256i*)(acc32.data()+j+8), _mm256_add_epi32(a1,hi32));
                }
#endif
                for (; j < col_cols; j++) {
                    int32_t s = 0;
                    for (int ci = pi; ci < end; ci++)
                        s += (int32_t)col_i8[pos[ci]*cc16+j];
                    acc32[j] += s;
                }
                pi = end;
            }

            // -1 weights: same but subtract
            int nni = 0;
            while (nni < (int)neg.size()) {
                int end = std::min(nni + FLUSH, (int)neg.size());
                int j = 0;
#ifdef __AVX2__
                for (; j+15 < cc16; j+=16) {
                    __m256i s16 = _mm256_setzero_si256();
                    for (int ci = nni; ci < end; ci++) {
                        __m128i b = _mm_loadu_si128((__m128i*)(col_i8.data() + neg[ci]*cc16 + j));
                        s16 = _mm256_add_epi16(s16, _mm256_cvtepi8_epi16(b));
                    }
                    __m128i lo16 = _mm256_castsi256_si128(s16);
                    __m128i hi16 = _mm256_extracti128_si256(s16, 1);
                    __m256i lo32 = _mm256_cvtepi16_epi32(lo16);
                    __m256i hi32 = _mm256_cvtepi16_epi32(hi16);
                    __m256i a0 = _mm256_loadu_si256((__m256i*)(acc32.data()+j));
                    __m256i a1 = _mm256_loadu_si256((__m256i*)(acc32.data()+j+8));
                    _mm256_storeu_si256((__m256i*)(acc32.data()+j), _mm256_sub_epi32(a0,lo32));
                    _mm256_storeu_si256((__m256i*)(acc32.data()+j+8), _mm256_sub_epi32(a1,hi32));
                }
#endif
                for (; j < col_cols; j++) {
                    int32_t s = 0;
                    for (int ci = nni; ci < end; ci++)
                        s += (int32_t)col_i8[neg[ci]*cc16+j];
                    acc32[j] -= s;
                }
                nni = end;
            }

            // Dequantize + fused scale + bias
            float* out_row = output.channel_ptr(ni, oc);
            float fs = layer.fused_scale[oc];
            float fb = layer.fused_bias[oc];
            float combined = inv_scale * fs;
            int j = 0;
#ifdef __AVX2__
            __m256 csv = _mm256_set1_ps(combined);
            __m256 bv = _mm256_set1_ps(fb);
            for (; j+7 < col_cols; j+=8) {
                __m256i iv = _mm256_loadu_si256((__m256i*)(acc32.data()+j));
                __m256 fv = _mm256_cvtepi32_ps(iv);
                _mm256_storeu_ps(out_row+j, _mm256_fmadd_ps(fv, csv, bv));
            }
#endif
            for (; j < col_cols; j++)
                out_row[j] = (float)acc32[j] * combined + fb;
        }
#if PROFILE_ENGINE
        g_prof.t_compute += pt.stop_ms();
        g_prof.ternary_calls++;
#endif
    }
    return output;
}

static void relu_inplace(Tensor& t){
    PTimer pt;pt.start();int sz=t.total();int i=0;
#ifdef __AVX2__
    __m256 z=_mm256_setzero_ps();
    for(;i+7<sz;i+=8){__m256 v=_mm256_loadu_ps(t.data.data()+i);_mm256_storeu_ps(t.data.data()+i,_mm256_max_ps(v,z));}
#endif
    for(;i<sz;i++)if(t.data[i]<0)t.data[i]=0;
#if PROFILE_ENGINE
    g_prof.t_relu+=pt.stop_ms();
#endif
}

static void add_inplace(Tensor& a,const Tensor& b){
    PTimer pt;pt.start();int sz=a.total();int i=0;
#ifdef __AVX2__
    for(;i+7<sz;i+=8){__m256 va=_mm256_loadu_ps(a.data.data()+i);__m256 vb=_mm256_loadu_ps(b.data.data()+i);_mm256_storeu_ps(a.data.data()+i,_mm256_add_ps(va,vb));}
#endif
    for(;i<sz;i++)a.data[i]+=b.data[i];
#if PROFILE_ENGINE
    g_prof.t_add+=pt.stop_ms();
#endif
}

static Tensor global_avg_pool(const Tensor& input){
    PTimer pt;pt.start();
    Tensor output(input.n,input.c,1,1);int sp=input.h*input.w;float inv=1.0f/sp;
    for(int ni=0;ni<input.n;ni++)for(int ci=0;ci<input.c;ci++){
        const float* p2=input.channel_ptr(ni,ci);float s=0;int i=0;
#ifdef __AVX2__
        __m256 sv=_mm256_setzero_ps();for(;i+7<sp;i+=8)sv=_mm256_add_ps(sv,_mm256_loadu_ps(p2+i));
        __m128 h=_mm256_extractf128_ps(sv,1),l=_mm256_castps256_ps128(sv);__m128 ss=_mm_add_ps(l,h);
        ss=_mm_hadd_ps(ss,ss);ss=_mm_hadd_ps(ss,ss);s=_mm_cvtss_f32(ss);
#endif
        for(;i<sp;i++)s+=p2[i];output.at(ni,ci,0,0)=s*inv;}
#if PROFILE_ENGINE
    g_prof.t_pool+=pt.stop_ms();
#endif
    return output;
}

static Tensor fc_forward(const Tensor& input,const FCLayer& layer){
    PTimer pt;pt.start();int batch=input.n;
    Tensor output(batch,layer.out_features,1,1);
    for(int ni=0;ni<batch;ni++){const float* ip=input.data.data()+ni*layer.in_features;
        for(uint32_t oc=0;oc<layer.out_features;oc++){
            const float* w=layer.weights.data()+oc*layer.in_features;
            float s=layer.bias[oc];int k=0;
#ifdef __AVX2__
            __m256 sv=_mm256_setzero_ps();for(;k+7<(int)layer.in_features;k+=8)
                sv=_mm256_fmadd_ps(_mm256_loadu_ps(w+k),_mm256_loadu_ps(ip+k),sv);
            __m128 h=_mm256_extractf128_ps(sv,1),l=_mm256_castps256_ps128(sv);__m128 ss=_mm_add_ps(l,h);
            ss=_mm_hadd_ps(ss,ss);ss=_mm_hadd_ps(ss,ss);s+=_mm_cvtss_f32(ss);
#endif
            for(;k<(int)layer.in_features;k++)s+=w[k]*ip[k];
            output.at(ni,oc,0,0)=s;}}
#if PROFILE_ENGINE
    g_prof.t_fc+=pt.stop_ms();
#endif
    return output;
}

class TernaryResNet18 {
public:
    Model model;
    std::vector<TernaryKernel> ternary_kernels;
    bool load(const std::string& path) {
        if(!model.load(path))return false;
        printf("Preparing V9 INT16 ternary kernels...\n");
        for(size_t i=0;i<model.layers.size();i++){
            TernaryKernel tk;
            if(model.layers[i].type==TERNARY_CONV){
                tk.prepare(model.layers[i].ternary_conv);
                auto& p=model.layers[i].ternary_conv.params;
                int tw=p.in_channels*p.kernel_h*p.kernel_w;
                float nz=0;for(uint32_t oc=0;oc<p.out_channels;oc++)
                    nz+=tk.pos_indices[oc].size()+tk.neg_indices[oc].size();
                nz/=p.out_channels;
                printf("  Layer %zu: %ux%u avg %.0f/%d nz (%.0f%% sparse) INT16 accum\n",
                       i,p.out_channels,p.in_channels,nz,tw,100*(1-nz/tw));
            }
            ternary_kernels.push_back(tk);
        }
        printf("V9 INT16 kernels ready.\n\n");return true;
    }
    Tensor forward(const Tensor& input){
        int idx=0;
        Tensor x=conv2d_fp32(input,model.layers[idx++].fp32_conv);relu_inplace(x);
        x=basic_block(x,idx,false);x=basic_block(x,idx,false);
        x=basic_block(x,idx,true);x=basic_block(x,idx,false);
        x=basic_block(x,idx,true);x=basic_block(x,idx,false);
        x=basic_block(x,idx,true);x=basic_block(x,idx,false);
        x=global_avg_pool(x);x=fc_forward(x,model.layers[idx++].fc);return x;
    }
    void print_profile(){g_prof.print();}
    void reset_profile(){g_prof.reset();}
private:
    Tensor basic_block(Tensor& x,int& idx,bool has_sc){
        Tensor id=x;
        Tensor out=conv2d_ternary_int16(x,model.layers[idx].ternary_conv,ternary_kernels[idx]);idx++;
        relu_inplace(out);
        out=conv2d_ternary_int16(out,model.layers[idx].ternary_conv,ternary_kernels[idx]);idx++;
        if(has_sc)id=conv2d_fp32(id,model.layers[idx++].fp32_conv);
        add_inplace(out,id);relu_inplace(out);return out;
    }
};