#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <immintrin.h>
#include <omp.h>
#include "model_loader.h"

// ═══════════════════════════════════════════════════════════════
// V10: 6-OC × 16-N Register-Tiled VNNI (MLAS-inspired)
//
// Key insight from disassembling ONNX Runtime's MLAS kernel:
//   MLAS keeps ALL accumulators in YMM registers (ymm4-15).
//   No load/store per dpbusd — eliminates store-forwarding stall.
//
// V9b had 8-OC tiling with stack accumulators:
//   load acc[I]+j → dpbusd → store acc[I]+j (L1 stall)
//
// V10: 6 OCs × 16 positions = 12 YMM register accumulators.
//   ymm0,1  = activation loads (16 positions)
//   ymm2    = weight broadcast (reused per OC)
//   ymm3    = spare
//   ymm4-15 = 12 accumulators (6 OCs × 2 position groups)
//
// Loop order flipped: j outer (16-pos blocks), g inner (K dim).
// Accumulators stay in registers for entire K loop.
// ═══════════════════════════════════════════════════════════════

#define PROFILE_ENGINE 1

struct EngineProfile {
    double t_quantize=0,t_im2col=0,t_compute=0;
    int ternary_calls=0;
    double t_fp32_total=0; int fp32_calls=0;
    double t_relu=0,t_add=0,t_pool=0,t_fc=0;
    void print(){
        double t_tern=t_quantize+t_im2col+t_compute;
        double total=t_tern+t_fp32_total+t_relu+t_add+t_pool+t_fc;
        printf("\n═══════════════════════════════════════════════════\n");
        printf("ENGINE PROFILE (V10, 6-OC×16-N register-tiled)\n");
        printf("═══════════════════════════════════════════════════\n\n");
        printf("Ternary Conv (%d calls):           %8.3f ms\n",ternary_calls,t_tern);
        printf("  ├─ quantize/pad input:           %8.3f ms  (%4.1f%%)\n",t_quantize,100*t_quantize/total);
        printf("  ├─ INT8 im2col + interleave:     %8.3f ms  (%4.1f%%)\n",t_im2col,100*t_im2col/total);
        printf("  └─ VNNI compute:                 %8.3f ms  (%4.1f%%)\n",t_compute,100*t_compute/total);
        printf("\nFP32 Conv (%d calls):              %8.3f ms  (%4.1f%%)\n",fp32_calls,t_fp32_total,100*t_fp32_total/total);
        printf("\nOther:\n");
        printf("  ├─ ReLU:                         %8.3f ms  (%4.1f%%)\n",t_relu,100*t_relu/total);
        printf("  ├─ Add:                          %8.3f ms  (%4.1f%%)\n",t_add,100*t_add/total);
        printf("  ├─ Pool:                         %8.3f ms  (%4.1f%%)\n",t_pool,100*t_pool/total);
        printf("  └─ FC:                           %8.3f ms  (%4.1f%%)\n",t_fc,100*t_fc/total);
        printf("\n───────────────────────────────────────────────────\n");
        printf("TOTAL:                             %8.3f ms\n",total);
        printf("═══════════════════════════════════════════════════\n");
    }
    void reset(){*this=EngineProfile{};}
};
static EngineProfile g_prof;
struct PTimer {
    std::chrono::high_resolution_clock::time_point s;
    void start(){s=std::chrono::high_resolution_clock::now();}
    double stop_ms(){return std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-s).count();}
};

struct Tensor {
    std::vector<float> data;
    int n,c,h,w;
    Tensor():n(0),c(0),h(0),w(0){}
    Tensor(int n,int c,int h,int w):data(n*c*h*w,0.0f),n(n),c(c),h(h),w(w){}
    void reshape(int nn,int nc,int nh,int nw){n=nn;c=nc;h=nh;w=nw;
        size_t need=(size_t)nn*nc*nh*nw; if(data.size()<need) data.resize(need);}
    float& at(int ni,int ci,int hi,int wi){return data[((ni*c+ci)*h+hi)*w+wi];}
    const float& at(int ni,int ci,int hi,int wi)const{return data[((ni*c+ci)*h+hi)*w+wi];}
    float* channel_ptr(int ni,int ci){return data.data()+((ni*c+ci)*h)*w;}
    const float* channel_ptr(int ni,int ci)const{return data.data()+((ni*c+ci)*h)*w;}
    int total()const{return n*c*h*w;}
};

struct INT8Buffer {
    std::vector<int8_t> data;
    int c,h,w; float inv_scale;
    INT8Buffer():c(0),h(0),w(0),inv_scale(0){}
    void reshape(int nc,int nh,int nw){c=nc;h=nh;w=nw;
        size_t need=(size_t)nc*nh*nw; if(data.size()<need) data.resize(need);}
};

// ─── FP32 conv ───────────────────────────────────────────────
// static void im2col_fp32(const Tensor& input,int ni,int kH,int kW,int stride,int padding,int out_h,int out_w,float* col){
//     int in_c=input.c,in_h=input.h,in_w=input.w;int idx=0;
//     for(int ic=0;ic<in_c;ic++){const float* ch=input.channel_ptr(ni,ic);
//         for(int kh=0;kh<kH;kh++)for(int kw=0;kw<kW;kw++)
//             for(int oh=0;oh<out_h;oh++){int ih=oh*stride-padding+kh;
//                 for(int ow=0;ow<out_w;ow++){int iw=ow*stride-padding+kw;
//                     col[idx++]=(ih>=0&&ih<in_h&&iw>=0&&iw<in_w)?ch[ih*in_w+iw]:0.0f;}}}
// }

static void im2col_fp32(
    const Tensor& input,
    int ni,
    int kH,
    int kW,
    int stride,
    int padding,
    int out_h,
    int out_w,
    float* __restrict col)
{
    const int in_c = input.c;
    const int in_h = input.h;
    const int in_w = input.w;

    int idx = 0;

    for (int ic = 0; ic < in_c; ic++) {
        const float* __restrict ch = input.channel_ptr(ni, ic);

        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {

                for (int oh = 0; oh < out_h; oh++) {
                    const int ih = oh * stride - padding + kh;

                    // whole row invalid → fill zeros
                    if (ih < 0 || ih >= in_h) {
                        std::memset(col + idx, 0, out_w * sizeof(float));
                        idx += out_w;
                        continue;
                    }

                    const float* row = ch + ih * in_w;

                    const int iw_start = -padding + kw;
                    const int iw_end   = iw_start + stride * (out_w - 1);

                    // fast path: contiguous valid row
                    if (stride == 1 &&
                        iw_start >= 0 &&
                        iw_end < in_w) {
                        std::memcpy(
                            col + idx,
                            row + iw_start,
                            out_w * sizeof(float));
                        idx += out_w;
                        continue;
                    }

                    // generic fallback
                    int iw = iw_start;
                    for (int ow = 0; ow < out_w; ow++, iw += stride) {
                        col[idx++] =
                            (iw >= 0 && iw < in_w) ? row[iw] : 0.0f;
                    }
                }
            }
        }
    }
}

static void conv2d_fp32_into(const Tensor& input,const FP32ConvLayer& layer,
                              std::vector<float>& col,Tensor& output){
    PTimer pt;auto& p=layer.params;
    int oh=(input.h+2*p.padding-p.kernel_h)/p.stride+1;
    int ow=(input.w+2*p.padding-p.kernel_w)/p.stride+1;
    output.reshape(input.n,p.out_channels,oh,ow);
    int cr=p.in_channels*p.kernel_h*p.kernel_w,cc=oh*ow;
    pt.start();
    for(int ni=0;ni<input.n;ni++){
        im2col_fp32(input,ni,p.kernel_h,p.kernel_w,p.stride,p.padding,oh,ow,col.data());
        for(uint32_t oc=0;oc<p.out_channels;oc++){
            const float* wr=layer.weights.data()+oc*cr;
            float* out=output.channel_ptr(ni,oc);
            float s=layer.fused_scale.empty()?1.0f:layer.fused_scale[oc];
            float b=layer.fused_bias.empty()?0.0f:layer.fused_bias[oc];
            memset(out,0,cc*sizeof(float));
            for(int k=0;k<cr;k++){
                const float* cr2=col.data()+k*cc;
                float wk=wr[k];
                int j=0;
#ifdef __AVX2__
                __m256 wv=_mm256_set1_ps(wk);
                for(;j+7<cc;j+=8){
                    __m256 a=_mm256_loadu_ps(out+j);
                    a=_mm256_fmadd_ps(wv,_mm256_loadu_ps(cr2+j),a);
                    _mm256_storeu_ps(out+j,a);
                }
#endif
                for(;j<cc;j++)
                    out[j]+=wk*cr2[j];
                }
            int j=0;
#ifdef __AVX2__
            __m256 sv=_mm256_set1_ps(s),bv=_mm256_set1_ps(b);
            for(;j+7<cc;j+=8){
                __m256 v=_mm256_loadu_ps(out+j);
                _mm256_storeu_ps(out+j,_mm256_fmadd_ps(v,sv,bv));
            }
#endif
            for(;j<cc;j++)
                out[j]=s*out[j]+b;
        }
    }
#if PROFILE_ENGINE
    g_prof.t_fp32_total+=pt.stop_ms();g_prof.fp32_calls++;
#endif
}

// static void conv2d_fp32_into(
//     const Tensor& input,
//     const FP32ConvLayer& layer,
//     std::vector<float>& col,
//     Tensor& output)
// {
//     PTimer pt;
//     const auto& p = layer.params;

//     const int oh = (input.h + 2 * p.padding - p.kernel_h) / p.stride + 1;
//     const int ow = (input.w + 2 * p.padding - p.kernel_w) / p.stride + 1;

//     output.reshape(input.n, p.out_channels, oh, ow);

//     const int cr = p.in_channels * p.kernel_h * p.kernel_w;
//     const int cc = oh * ow;

//     pt.start();

//     for (int ni = 0; ni < input.n; ++ni) {
//         im2col_fp32(
//             input, ni,
//             p.kernel_h, p.kernel_w,
//             p.stride, p.padding,
//             oh, ow,
//             col.data());

//         for (uint32_t oc = 0; oc < p.out_channels; ++oc) {
//             const float* wr = layer.weights.data() + oc * cr;
//             float* out = output.channel_ptr(ni, oc);

//             const float s = layer.fused_scale.empty() ? 1.0f : layer.fused_scale[oc];
//             const float b = layer.fused_bias.empty() ? 0.0f : layer.fused_bias[oc];

//             int j = 0;

// #ifdef __AVX2__
//             const __m256 sv = _mm256_set1_ps(s);
//             const __m256 bv = _mm256_set1_ps(b);

//             for (; j + 7 < cc; j += 8) {
//                 __m256 acc = _mm256_setzero_ps();

//                 for (int k = 0; k < cr; ++k) {
//                     const __m256 xv = _mm256_loadu_ps(col.data() + k * cc + j);
//                     const __m256 wv = _mm256_set1_ps(wr[k]);
//                     acc = _mm256_fmadd_ps(wv, xv, acc);
//                 }

//                 acc = _mm256_fmadd_ps(acc, sv, bv);
//                 _mm256_storeu_ps(out + j, acc);
//             }
// #endif

//             // scalar tail
//             for (; j < cc; ++j) {
//                 float acc = 0.0f;
//                 for (int k = 0; k < cr; ++k) {
//                     acc += wr[k] * col[k * cc + j];
//                 }
//                 out[j] = s * acc + b;
//             }
//         }
//     }

// #if PROFILE_ENGINE
//     g_prof.t_fp32_total += pt.stop_ms();
//     g_prof.fp32_calls++;
// #endif
// }


// ─── TernaryKernel ───────────────────────────────────────────
struct TernaryKernel {
    std::vector<uint8_t> packed_weights;
    std::vector<int8_t>  unpacked_weights;  // NHWC K-order: [KH][KW][IC]
    std::vector<int32_t> weight_sum;
    uint32_t out_channels;
    int col_rows,packed_bytes,col_rows_pad4;
    int in_channels, kH, kW;  // stored for NHWC im2col

    void prepare(const TernaryConvLayer& layer){
        auto& p=layer.params;
        out_channels=p.out_channels;
        in_channels=p.in_channels;
        kH=p.kernel_h; kW=p.kernel_w;
        col_rows=p.in_channels*p.kernel_h*p.kernel_w;
        packed_bytes=(col_rows+3)/4;
        col_rows_pad4=((col_rows+3)/4)*4;

        // First unpack to NCHW order in a temp buffer
        std::vector<int8_t> uw_nchw(out_channels*col_rows_pad4,0);
        packed_weights.resize(out_channels*packed_bytes,0x55);
        for(uint32_t oc=0;oc<out_channels;oc++){
            uint8_t* pw=packed_weights.data()+oc*packed_bytes;
            int bo=oc*col_rows;
            for(int k=0;k<col_rows;k++){
                int bi=bo+k,byte_idx=k/4,bit_pos=(k%4)*2;
                uint8_t val=0x01;
                if((layer.mask_pos[bi>>3]>>(bi&7))&1) val=0x02;
                else if((layer.mask_neg[bi>>3]>>(bi&7))&1) val=0x00;
                pw[byte_idx]=(pw[byte_idx]&~(0x03<<bit_pos))|(val<<bit_pos);
            }
        }
        weight_sum.resize(out_channels,0);
        for(uint32_t oc=0;oc<out_channels;oc++){
            const uint8_t* pw2=packed_weights.data()+oc*packed_bytes;
            int8_t* uw=uw_nchw.data()+oc*col_rows_pad4;
            int32_t wsum=0;
            for(int k=0;k<col_rows;k++){
                int byte_idx=k/4,bit_pos=(k%4)*2;
                int8_t w=(int8_t)(((pw2[byte_idx]>>bit_pos)&0x03)-1);
                uw[k]=w;wsum+=w;
            }
            weight_sum[oc]=wsum;
        }

        // Reorder to NHWC K-order: [KH][KW][IC] (was [IC][KH][KW])
        // This makes 4 consecutive K-values = 4 contiguous IC channels at same (kh,kw)
        int IC=(int)p.in_channels, KH=(int)p.kernel_h, KW2=(int)p.kernel_w;
        unpacked_weights.resize(out_channels*col_rows_pad4,0);
        for(uint32_t oc=0;oc<out_channels;oc++){
            for(int ic=0;ic<IC;ic++){
                for(int kh=0;kh<KH;kh++){
                    for(int kw=0;kw<KW2;kw++){
                        int nchw_k = ic*KH*KW2 + kh*KW2 + kw;
                        int nhwc_k = (kh*KW2 + kw)*IC + ic;
                        unpacked_weights[oc*col_rows_pad4 + nhwc_k] =
                            uw_nchw[oc*col_rows_pad4 + nchw_k];
                    }
                }
            }
        }
    }
};

// ─── NCHW→NHWC transpose (s8→u8 with +128) ──────────────────
// Transpose C×H×W s8 into padded [pH][pW][C] u8
static void transpose_nchw_s8_to_nhwc_u8(const int8_t* nchw, int C, int H, int W,
                                           int padding, uint8_t* nhwc, int pH, int pW) {
    // Fill padding with zero-point (128)
    memset(nhwc, 128, (size_t)pH * pW * C);

    // Transpose with write-sequential order: (h, w, c)
    // Writes are sequential (stride 1), reads jump by H*W per channel
    for (int ih = 0; ih < H; ih++) {
        for (int iw = 0; iw < W; iw++) {
            uint8_t* dst = nhwc + ((ih + padding) * pW + (iw + padding)) * C;
            int src_offset = ih * W + iw;
            int ic = 0;
#ifdef __AVX2__
            __m256i bias = _mm256_set1_epi8((char)0x80);
            int hw = H * W;
            for (; ic + 31 < C; ic += 32) {
                // Gather 32 s8 values from 32 channels (strided by H*W)
                // Use scalar loads — gather is slower on most CPUs
                alignas(32) int8_t tmp[32];
                for (int k = 0; k < 32; k++)
                    tmp[k] = nchw[(ic + k) * hw + src_offset];
                __m256i v = _mm256_load_si256((const __m256i*)tmp);
                _mm256_storeu_si256((__m256i*)(dst + ic),
                                   _mm256_xor_si256(v, bias));
            }
#endif
            for (; ic < C; ic++)
                dst[ic] = (uint8_t)((int)nchw[ic * H * W + src_offset] + 128);
        }
    }
}

// ─── Quantize FP32 NCHW → padded u8 NHWC ────────────────────
// Two-pass: (1) quantize to NCHW s8 (fast sequential), (2) transpose to NHWC u8
static float quantize_fp32_to_nhwc_u8(const Tensor& input, int ni, int padding,
                                       uint8_t* nhwc_padded, int pH, int pW,
                                       int8_t* temp_nchw) {
    int C=input.c, H=input.h, W=input.w;
    int tot=C*H*W;
    const float* src=input.data.data()+ni*tot;

    // Pass 1: find absmax + quantize to NCHW s8 (sequential access)
    float gmax=0.0f; int i=0;
#ifdef __AVX2__
    __m256 vm=_mm256_setzero_ps(),sm=_mm256_set1_ps(-0.0f);
    for(;i+7<tot;i+=8){__m256 v=_mm256_andnot_ps(sm,_mm256_loadu_ps(src+i));vm=_mm256_max_ps(vm,v);}
    __m128 hh=_mm256_extractf128_ps(vm,1),ll=_mm256_castps256_ps128(vm);
    __m128 m=_mm_max_ps(ll,hh);m=_mm_max_ps(m,_mm_shuffle_ps(m,m,_MM_SHUFFLE(2,3,0,1)));
    m=_mm_max_ps(m,_mm_shuffle_ps(m,m,_MM_SHUFFLE(1,0,3,2)));gmax=_mm_cvtss_f32(m);
#endif
    for(;i<tot;i++){float v=fabsf(src[i]);if(v>gmax)gmax=v;}
    if(gmax<1e-10f){
        memset(nhwc_padded, 128, (size_t)pH*pW*C);
        return 0.0f;
    }

    float qscale=127.0f/gmax;
    i=0;
#ifdef __AVX2__
    __m256 qv=_mm256_set1_ps(qscale);
    for(;i+7<tot;i+=8){
        __m256 v=_mm256_mul_ps(_mm256_loadu_ps(src+i),qv);
        __m256i i32=_mm256_cvtps_epi32(v);
        __m256i p16=_mm256_packs_epi32(i32,i32);
        p16=_mm256_permute4x64_epi64(p16,0xD8);
        _mm_storel_epi64((__m128i*)(temp_nchw+i),
            _mm_packs_epi16(_mm256_castsi256_si128(p16),_mm256_castsi256_si128(p16)));
    }
#endif
    for(;i<tot;i++){
        int v=(int)roundf(src[i]*qscale);
        temp_nchw[i]=(int8_t)(v>127?127:(v<-128?-128:v));
    }

    // Pass 2: transpose NCHW s8 → padded NHWC u8
    transpose_nchw_s8_to_nhwc_u8(temp_nchw, C, H, W, padding,
                                  nhwc_padded, pH, pW);
    return gmax/127.0f;
}

// ─── Convert s8 INT8Buffer → padded u8 NHWC ─────────────────
static float pad_int8_to_nhwc_u8(const INT8Buffer& buf, int padding,
                                  uint8_t* nhwc_padded, int pH, int pW) {
    transpose_nchw_s8_to_nhwc_u8(buf.data.data(), buf.c, buf.h, buf.w,
                                  padding, nhwc_padded, pH, pW);
    return buf.inv_scale;
}

// ─── NHWC im2col + VNNI pack (fused, no separate interleave) ─
// Reads from padded u8 NHWC [pH][pW][C] input.
// Weights use NHWC K-order: [KH][KW][IC], so group g = 4 consecutive IC
// channels from the same (kh,kw) → 4 contiguous bytes in NHWC input.
// Output: col_vnni[g * cc16 + j] (4 bytes), already u8.
static void nhwc_im2col_pack(const uint8_t* nhwc_padded, int C,
                              int pH, int pW,
                              int kH, int kW, int stride,
                              int out_h, int out_w,
                              uint8_t* col_vnni, int cc16) {
    int col_cols = out_h * out_w;
    int num_groups = ((C * kH * kW) + 3) / 4;

    for (int kh = 0; kh < kH; kh++) {
        for (int kw = 0; kw < kW; kw++) {
            int spatial_idx = kh * kW + kw;
            int ic_groups = C / 4;  // IC is always multiple of 4

            for (int icg = 0; icg < ic_groups; icg++) {
                int g = spatial_idx * ic_groups + icg;
                int ic_start = icg * 4;
                uint8_t* dst_g = col_vnni + g * cc16 * 4;

                for (int oh = 0; oh < out_h; oh++) {
                    int ih = oh * stride + kh;
                    for (int ow = 0; ow < out_w; ow++) {
                        int iw = ow * stride + kw;
                        int j = oh * out_w + ow;
                        // 4 contiguous bytes in NHWC input!
                        const uint8_t* src = nhwc_padded +
                            (ih * pW + iw) * C + ic_start;
                        *(uint32_t*)(dst_g + j * 4) = *(const uint32_t*)src;
                    }
                }
                // Zero-pad remaining positions
                for (int j = col_cols; j < cc16; j++)
                    *(uint32_t*)(dst_g + j * 4) = 0x80808080u; // u8 zero-point
            }
        }
    }
}
// static void interleave_vnni(const int8_t* col_i8,int col_rows,int cc16,uint8_t* col_vnni){
//     int num_groups=col_rows/4,remaining=col_rows-num_groups*4;
// #ifdef __AVX2__
//     __m256i xor_mask=_mm256_set1_epi8((char)0x80);
//     for(int g=0;g<num_groups;g++){
//         const int8_t* r0=col_i8+(g*4+0)*cc16,*r1=col_i8+(g*4+1)*cc16;
//         const int8_t* r2=col_i8+(g*4+2)*cc16,*r3=col_i8+(g*4+3)*cc16;
//         uint8_t* dst=col_vnni+g*cc16*4;int j=0;
//         for(;j+7<cc16;j+=8){
//             __m128i v0=_mm_loadl_epi64((__m128i*)(r0+j)),v1=_mm_loadl_epi64((__m128i*)(r1+j));
//             __m128i v2=_mm_loadl_epi64((__m128i*)(r2+j)),v3=_mm_loadl_epi64((__m128i*)(r3+j));
//             __m128i t01=_mm_unpacklo_epi8(v0,v1),t23=_mm_unpacklo_epi8(v2,v3);
//             _mm256_storeu_si256((__m256i*)(dst+j*4),_mm256_xor_si256(
//                 _mm256_setr_m128i(_mm_unpacklo_epi16(t01,t23),_mm_unpackhi_epi16(t01,t23)),xor_mask));
//         }
//         for(;j<cc16;j++){dst[j*4+0]=(uint8_t)((int)r0[j]+128);dst[j*4+1]=(uint8_t)((int)r1[j]+128);
//             dst[j*4+2]=(uint8_t)((int)r2[j]+128);dst[j*4+3]=(uint8_t)((int)r3[j]+128);}
//     }
// #else
//     for(int g=0;g<num_groups;g++){uint8_t* dst=col_vnni+g*cc16*4;
//         for(int j=0;j<cc16;j++)for(int r=0;r<4;r++) dst[j*4+r]=(uint8_t)((int)col_i8[(g*4+r)*cc16+j]+128);}
// #endif
//     if(remaining>0){uint8_t* dst=col_vnni+num_groups*cc16*4;
//         for(int j=0;j<cc16;j++)for(int r=0;r<4;r++){int k=num_groups*4+r;
//             dst[j*4+r]=(k<col_rows)?(uint8_t)((int)col_i8[k*cc16+j]+128):128;}}
// }



static void interleave_vnni(
    const int8_t* __restrict col_i8,
    int col_rows,
    int cc16,
    uint8_t* __restrict dst)
{
    const int num_groups = (col_rows + 3) / 4;
    const __m256i bias = _mm256_set1_epi8(char(0x80));

    for (int g = 0; g < num_groups; g++) {
        const int8_t* r0 = col_i8 + (g * 4 + 0) * cc16;
        const int8_t* r1 = col_i8 + (g * 4 + 1) * cc16;
        const int8_t* r2 = col_i8 + (g * 4 + 2) * cc16;
        const int8_t* r3 = col_i8 + (g * 4 + 3) * cc16;

        uint8_t* out = dst + g * cc16 * 4;

        int j = 0;

        for (; j + 31 < cc16; j += 32) {
            __m256i v0 = _mm256_loadu_si256((const __m256i*)(r0 + j));
            __m256i v1 = _mm256_loadu_si256((const __m256i*)(r1 + j));
            __m256i v2 = _mm256_loadu_si256((const __m256i*)(r2 + j));
            __m256i v3 = _mm256_loadu_si256((const __m256i*)(r3 + j));

            // convert s8 -> u8 by flipping sign bit
            v0 = _mm256_xor_si256(v0, bias);
            v1 = _mm256_xor_si256(v1, bias);
            v2 = _mm256_xor_si256(v2, bias);
            v3 = _mm256_xor_si256(v3, bias);

            // interleave pairs
            __m256i t0 = _mm256_unpacklo_epi8(v0, v1);
            __m256i t1 = _mm256_unpackhi_epi8(v0, v1);
            __m256i t2 = _mm256_unpacklo_epi8(v2, v3);
            __m256i t3 = _mm256_unpackhi_epi8(v2, v3);

            // interleave 4 rows
            __m256i o0 = _mm256_unpacklo_epi16(t0, t2);
            __m256i o1 = _mm256_unpackhi_epi16(t0, t2);
            __m256i o2 = _mm256_unpacklo_epi16(t1, t3);
            __m256i o3 = _mm256_unpackhi_epi16(t1, t3);

            _mm256_storeu_si256((__m256i*)(out + (j +  0) * 4), o0);
            _mm256_storeu_si256((__m256i*)(out + (j +  8) * 4), o1);
            _mm256_storeu_si256((__m256i*)(out + (j + 16) * 4), o2);
            _mm256_storeu_si256((__m256i*)(out + (j + 24) * 4), o3);
        }

        // scalar tail
        for (; j < cc16; j++) {
            out[j * 4 + 0] = uint8_t(r0[j] ^ 0x80);
            out[j * 4 + 1] = uint8_t(r1[j] ^ 0x80);
            out[j * 4 + 2] = uint8_t(r2[j] ^ 0x80);
            out[j * 4 + 3] = uint8_t(r3[j] ^ 0x80);
        }
    }
}




// ═══════════════════════════════════════════════════════════════
// 8-OC TILED VNNI COMPUTE KERNEL
//
// Process 8 output channels per activation load.
// 8 independent load-dpbusd-store chains eliminate the
// store-forwarding stall (L1 Latency Dependency = 19.7%)
// and amortize loop overhead (Port 6 = 43.6%).
//
// Register usage per g-iteration:
//   8 × set1_epi32 weight broadcasts (hoisted outside j-loop)
//   1 × activation load (shared by all 8 OCs)
//   Accumulators live in stack memory (aligned int32_t arrays)
//   Each dpbusd: load acc → compute → store acc (transient)
// ═══════════════════════════════════════════════════════════════

// Helper: compute + dequant for one OC (inlined by compiler)
static inline void dequant_oc(const int32_t* acc, float* out_row,
                               const float* res_row, int col_cols,
                               float combined, float adj_bias, bool apply_relu) {
    int j = 0;
    __m256 vs = _mm256_set1_ps(combined);
    __m256 vb = _mm256_set1_ps(adj_bias);
    __m256 vz = _mm256_setzero_ps();
    for (; j + 7 < col_cols; j += 8) {
        __m256 v = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i*)(acc + j)));
        __m256 r = _mm256_fmadd_ps(v, vs, vb);
        if (res_row) r = _mm256_add_ps(r, _mm256_loadu_ps(res_row + j));
        if (apply_relu) r = _mm256_max_ps(r, vz);
        _mm256_storeu_ps(out_row + j, r);
    }
    for (; j < col_cols; j++) {
        float val = (float)acc[j] * combined + adj_bias;
        if (res_row) val += res_row[j];
        if (apply_relu) val = std::max(0.0f, val);
        out_row[j] = val;
    }
}


static void conv2d_ternary_vnni(
    const Tensor* input_fp32, const INT8Buffer* input_i8,
    const TernaryConvLayer& layer, TernaryKernel& kernel,
    std::vector<int8_t>& padded_buf, std::vector<int8_t>& col_i8,
    std::vector<uint8_t>& col_vnni,
    Tensor& output_fp32, INT8Buffer* output_i8,
    const Tensor* residual, bool apply_relu)
{
    PTimer pt;
    auto& p = layer.params;
    int in_h = input_fp32 ? input_fp32->h : input_i8->h;
    int in_w = input_fp32 ? input_fp32->w : input_i8->w;
    int in_c = input_fp32 ? input_fp32->c : input_i8->c;
    int batch = input_fp32 ? input_fp32->n : 1;
    int out_h = (in_h + 2*p.padding - p.kernel_h) / p.stride + 1;
    int out_w = (in_w + 2*p.padding - p.kernel_w) / p.stride + 1;
    int col_rows = kernel.col_rows, col_cols = out_h * out_w;
    int cc16 = ((col_cols + 15) / 16) * 16;
    int pH = in_h + 2*p.padding, pW = in_w + 2*p.padding;
    int num_groups = kernel.col_rows_pad4 / 4;

    output_fp32.reshape(batch, p.out_channels, out_h, out_w);
    if (output_i8) output_i8->reshape(p.out_channels, out_h, out_w);

    for (int ni = 0; ni < batch; ni++) {
        float inv_scale;

        // ── Step 1: Quantize to padded NHWC u8 ──
        pt.start();
        if (input_fp32) {
            inv_scale = quantize_fp32_to_nhwc_u8(*input_fp32, ni, p.padding,
                                                  (uint8_t*)padded_buf.data(), pH, pW,
                                                  col_i8.data());  // reuse col_i8 as temp
        } else {
            inv_scale = pad_int8_to_nhwc_u8(*input_i8, p.padding,
                                             (uint8_t*)padded_buf.data(), pH, pW);
        }
#if PROFILE_ENGINE
        g_prof.t_quantize += pt.stop_ms();
#endif

        // ── Step 2: NHWC im2col + VNNI pack (fused, no interleave) ──
        pt.start();
        nhwc_im2col_pack((const uint8_t*)padded_buf.data(), in_c,
                          pH, pW, p.kernel_h, p.kernel_w, p.stride,
                          out_h, out_w, col_vnni.data(), cc16);
#if PROFILE_ENGINE
        g_prof.t_im2col += pt.stop_ms();
#endif

        // ── Step 3: VNNI compute ──
        // ═══════════════════════════════════════════════════════
        // V10: 6-OC × 16-N Register-Tiled VNNI
        //
        // MLAS-inspired: all 12 accumulators live in YMM registers.
        // Loop order: j outer (16-pos blocks) → g inner (K dim).
        // No load/store per dpbusd — eliminates store-forwarding.
        // Unified path for all cc16 values and output types.
        // ═══════════════════════════════════════════════════════
        pt.start();
        {
            const uint8_t* col_ptr = col_vnni.data();

            #pragma omp parallel for schedule(dynamic)
            for (uint32_t oc_base = 0; oc_base < p.out_channels; oc_base += 6) {
                uint32_t oc_end = std::min(oc_base + 6, p.out_channels);
                uint32_t oc_count = oc_end - oc_base;

                // Weight pointers (pad unused slots to avoid branches in hot loop)
                const int8_t* uw[6];
                for (uint32_t i = 0; i < oc_count; i++)
                    uw[i] = kernel.unpacked_weights.data()
                            + (oc_base + i) * kernel.col_rows_pad4;
                for (uint32_t i = oc_count; i < 6; i++)
                    uw[i] = uw[0]; // dummy — results ignored

                // Accumulator storage: written after each 16-pos register block
                alignas(32) int32_t acc_store[6][1024];

                for (int j = 0; j < cc16; j += 16) {
                    // ── 12 accumulator registers (6 OCs × 2 groups of 8) ──
                    __m256i r0a = _mm256_setzero_si256(), r0b = _mm256_setzero_si256();
                    __m256i r1a = _mm256_setzero_si256(), r1b = _mm256_setzero_si256();
                    __m256i r2a = _mm256_setzero_si256(), r2b = _mm256_setzero_si256();
                    __m256i r3a = _mm256_setzero_si256(), r3b = _mm256_setzero_si256();
                    __m256i r4a = _mm256_setzero_si256(), r4b = _mm256_setzero_si256();
                    __m256i r5a = _mm256_setzero_si256(), r5b = _mm256_setzero_si256();

                    // ── K loop: accumulators stay in registers ──
                    for (int g = 0; g < num_groups; g++) {
                        // Load 16 activation positions (2 × ymm)
                        __m256i act0 = _mm256_loadu_si256(
                            (const __m256i*)(col_ptr + (g * cc16 + j) * 4));
                        __m256i act1 = _mm256_loadu_si256(
                            (const __m256i*)(col_ptr + (g * cc16 + j + 8) * 4));

                        // 6 OCs: broadcast weight, 2× dpbusd per OC
                        __m256i w;

                        w = _mm256_set1_epi32(*(const int32_t*)(uw[0] + g * 4));
                        r0a = _mm256_dpbusd_avx_epi32(r0a, act0, w);
                        r0b = _mm256_dpbusd_avx_epi32(r0b, act1, w);

                        w = _mm256_set1_epi32(*(const int32_t*)(uw[1] + g * 4));
                        r1a = _mm256_dpbusd_avx_epi32(r1a, act0, w);
                        r1b = _mm256_dpbusd_avx_epi32(r1b, act1, w);

                        w = _mm256_set1_epi32(*(const int32_t*)(uw[2] + g * 4));
                        r2a = _mm256_dpbusd_avx_epi32(r2a, act0, w);
                        r2b = _mm256_dpbusd_avx_epi32(r2b, act1, w);

                        w = _mm256_set1_epi32(*(const int32_t*)(uw[3] + g * 4));
                        r3a = _mm256_dpbusd_avx_epi32(r3a, act0, w);
                        r3b = _mm256_dpbusd_avx_epi32(r3b, act1, w);

                        w = _mm256_set1_epi32(*(const int32_t*)(uw[4] + g * 4));
                        r4a = _mm256_dpbusd_avx_epi32(r4a, act0, w);
                        r4b = _mm256_dpbusd_avx_epi32(r4b, act1, w);

                        w = _mm256_set1_epi32(*(const int32_t*)(uw[5] + g * 4));
                        r5a = _mm256_dpbusd_avx_epi32(r5a, act0, w);
                        r5b = _mm256_dpbusd_avx_epi32(r5b, act1, w);
                    }

                    // ── Store register accumulators to buffer ──
                    _mm256_storeu_si256((__m256i*)(acc_store[0] + j),     r0a);
                    _mm256_storeu_si256((__m256i*)(acc_store[0] + j + 8), r0b);
                    _mm256_storeu_si256((__m256i*)(acc_store[1] + j),     r1a);
                    _mm256_storeu_si256((__m256i*)(acc_store[1] + j + 8), r1b);
                    _mm256_storeu_si256((__m256i*)(acc_store[2] + j),     r2a);
                    _mm256_storeu_si256((__m256i*)(acc_store[2] + j + 8), r2b);
                    _mm256_storeu_si256((__m256i*)(acc_store[3] + j),     r3a);
                    _mm256_storeu_si256((__m256i*)(acc_store[3] + j + 8), r3b);
                    _mm256_storeu_si256((__m256i*)(acc_store[4] + j),     r4a);
                    _mm256_storeu_si256((__m256i*)(acc_store[4] + j + 8), r4b);
                    _mm256_storeu_si256((__m256i*)(acc_store[5] + j),     r5a);
                    _mm256_storeu_si256((__m256i*)(acc_store[5] + j + 8), r5b);
                }

                // ── Dequant all OCs in tile ──
                for (uint32_t i = 0; i < oc_count; i++) {
                    uint32_t oc = oc_base + i;
                    float combined = inv_scale * layer.fused_scale[oc];
                    float adj_bias = layer.fused_bias[oc]
                                     - 128.0f * (float)kernel.weight_sum[oc] * combined;
                    dequant_oc(acc_store[i], output_fp32.channel_ptr(ni, oc),
                              (!output_i8 && residual) ? residual->channel_ptr(ni, oc) : nullptr,
                              col_cols, combined, adj_bias, apply_relu);
                }
            }

            // ── INT8 output: requantize ──
            if (output_i8) {
                int total_el = p.out_channels * col_cols;
                const float* src = output_fp32.data.data();
                float gmax = 0.0f; int i = 0;
#ifdef __AVX2__
                __m256 vm=_mm256_setzero_ps(),sm=_mm256_set1_ps(-0.0f);
                for(;i+7<total_el;i+=8){__m256 v=_mm256_andnot_ps(sm,_mm256_loadu_ps(src+i));vm=_mm256_max_ps(vm,v);}
                __m128 hh=_mm256_extractf128_ps(vm,1),ll=_mm256_castps256_ps128(vm);
                __m128 mm=_mm_max_ps(ll,hh);mm=_mm_max_ps(mm,_mm_shuffle_ps(mm,mm,_MM_SHUFFLE(2,3,0,1)));
                mm=_mm_max_ps(mm,_mm_shuffle_ps(mm,mm,_MM_SHUFFLE(1,0,3,2)));gmax=_mm_cvtss_f32(mm);
#endif
                for(;i<total_el;i++){float v=fabsf(src[i]);if(v>gmax)gmax=v;}
                if(gmax<1e-10f){memset(output_i8->data.data(),0,(size_t)total_el);output_i8->inv_scale=0.0f;}
                else{
                    float qscale=127.0f/gmax;output_i8->inv_scale=gmax/127.0f;
                    int8_t* dst=output_i8->data.data();i=0;
#ifdef __AVX2__
                    __m256 qv=_mm256_set1_ps(qscale);
                    for(;i+7<total_el;i+=8){__m256 v=_mm256_mul_ps(_mm256_loadu_ps(src+i),qv);
                        __m256i i32=_mm256_cvtps_epi32(v);__m256i p16=_mm256_packs_epi32(i32,i32);
                        p16=_mm256_permute4x64_epi64(p16,0xD8);
                        _mm_storel_epi64((__m128i*)(dst+i),_mm_packs_epi16(_mm256_castsi256_si128(p16),_mm256_castsi256_si128(p16)));}
#endif
                    for(;i<total_el;i++){int v=(int)roundf(src[i]*qscale);dst[i]=(int8_t)(v>127?127:(v<-128?-128:v));}
                }
            }
        }

#if PROFILE_ENGINE
        g_prof.t_compute += pt.stop_ms();
        g_prof.ternary_calls++;
#endif
    }
}


// ─── Standard ops ────────────────────────────────────────────
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


// ═══════════════════════════════════════════════════════════════
// TernaryResNet18 — zero-alloc, 8-OC tiled
// ═══════════════════════════════════════════════════════════════

class TernaryResNet18 {
public:
    Model model;
    std::vector<TernaryKernel> ternary_kernels;
    std::vector<float>   col_fp32_buf;
    std::vector<int8_t>  padded_buf;
    std::vector<int8_t>  col_i8_buf;
    std::vector<uint8_t> col_vnni_buf;
    Tensor work_a, work_b, work_sc;
    INT8Buffer int8_mid;

    bool load(const std::string& path, const std::string& /*calib_path*/="") {
        if (!model.load(path)) return false;
        printf("Preparing V10 (6-OC×16-N register-tiled) kernels...\n");
        int max_fp32_col=0, max_padded=0, max_col_i8=0, max_col_vnni=0, max_out=0;
        size_t total_packed=0;
        for (size_t i = 0; i < model.layers.size(); i++) {
            TernaryKernel tk;
            if (model.layers[i].type == TERNARY_CONV) {
                tk.prepare(model.layers[i].ternary_conv);
                auto& p2 = model.layers[i].ternary_conv.params;
                int cr=tk.col_rows, pH=32+2*p2.padding, pW=32+2*p2.padding;
                int padded_sz=p2.in_channels*pH*pW, cc16=((32*32+15)/16)*16;
                int col_sz=cr*cc16, num_g=(cr+3)/4, vnni_sz=num_g*cc16*4;
                int out_el=p2.out_channels*32*32;
                if(padded_sz>max_padded) max_padded=padded_sz;
                if(col_sz>max_col_i8) max_col_i8=col_sz;
                if(vnni_sz>max_col_vnni) max_col_vnni=vnni_sz;
                if(out_el>max_out) max_out=out_el;
                total_packed += p2.out_channels*tk.packed_bytes;
                printf("  Layer %zu: %ux%u K=%d packed=%d B/filt (%d KB)\n",
                       i, p2.out_channels, p2.in_channels, tk.col_rows,
                       tk.packed_bytes, (int)(p2.out_channels*tk.packed_bytes/1024));
            }
            if (model.layers[i].type == FP32_CONV) {
                auto& p2=model.layers[i].fp32_conv.params;
                int cr=p2.in_channels*p2.kernel_h*p2.kernel_w, cc=32*32;
                if(cr*cc>max_fp32_col) max_fp32_col=cr*cc;
            }
            ternary_kernels.push_back(tk);
        }
        col_fp32_buf.resize(max_fp32_col); padded_buf.resize(max_padded);
        col_i8_buf.resize(max_col_i8); col_vnni_buf.resize(max_col_vnni);
        work_a.data.resize(max_out); work_b.data.resize(max_out);
        work_sc.data.resize(max_out); int8_mid.data.resize(max_out);

        printf("I2S packed weights: %.1f KB\n", total_packed/1024.0);
        printf("Pre-allocated: padded=%.1fKB col_i8=%.1fKB col_vnni=%.1fKB\n",
               max_padded/1024.0, max_col_i8/1024.0, max_col_vnni/1024.0);
        printf("V10 (6-OC×16-N register-tiled) ready.\n\n");
        return true;
    }

    Tensor forward(const Tensor& input) {
        int idx = 0;
        conv2d_fp32_into(input, model.layers[idx++].fp32_conv, col_fp32_buf, work_a);
        relu_inplace(work_a);
        basic_block(work_a,work_b,idx,false); basic_block(work_b,work_a,idx,false);
        basic_block(work_a,work_b,idx,true);  basic_block(work_b,work_a,idx,false);
        basic_block(work_a,work_b,idx,true);  basic_block(work_b,work_a,idx,false);
        basic_block(work_a,work_b,idx,true);  basic_block(work_b,work_a,idx,false);
        Tensor pooled = global_avg_pool(work_a);
        return fc_forward(pooled, model.layers[idx++].fc);
    }
    void print_profile() { g_prof.print(); }
    void reset_profile() { g_prof.reset(); }

private:
    void basic_block(Tensor& x_in, Tensor& x_out, int& idx, bool has_sc) {
        int idx1=idx++, idx2=idx++, idx_sc=has_sc?idx++:-1;

        conv2d_ternary_vnni(&x_in, nullptr,
            model.layers[idx1].ternary_conv, ternary_kernels[idx1],
            padded_buf, col_i8_buf, col_vnni_buf,
            work_sc, &int8_mid, nullptr, true);

        if (has_sc)
            conv2d_fp32_into(x_in, model.layers[idx_sc].fp32_conv,
                            col_fp32_buf, work_sc);

        const Tensor* res = has_sc ? &work_sc : &x_in;
        conv2d_ternary_vnni(nullptr, &int8_mid,
            model.layers[idx2].ternary_conv, ternary_kernels[idx2],
            padded_buf, col_i8_buf, col_vnni_buf,
            x_out, nullptr, res, true);
    }
};