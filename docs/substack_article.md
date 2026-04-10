# What If Your Model Weights Were Just {-1, 0, +1}?

## 19 Experiments, Mostly Failures, One Fast Engine

*Shubham Saini*

---

I spent the last few weeks building a custom C++ inference engine for a ternary neural network. The weights are literally just negative one, zero, and plus one. No floating point. No 8-bit integers. Two bits per weight.

The result: a ResNet-18 inference engine that runs CIFAR-10 classification in **3.71ms single-threaded** and **0.99ms at 6 threads** on an Intel i5-13400F (Raptor Lake), with a model **12.5x smaller** than FP32 and **3.2x smaller** than ONNX INT8. It comfortably beats ONNX INT8 Dynamic (2.54ms at 6T) and comes within 13% of ONNX INT8 Static (0.88ms at 6T).

Getting there took 19 experiments. Most of them failed. This is the story of what I learned.

---

## The Promise and the Problem

Ternary weight networks are seductive on paper. If your weights are constrained to {-1, 0, +1}, you do not need multiplications at all. A convolution becomes a series of additions and subtractions. Your model compresses to 2 bits per weight. A 44.69 MB ResNet-18 shrinks to 3.56 MB. That is a 92.2% reduction.

The problem is that "theoretically efficient" and "actually fast" are not the same thing. I know this because my first ternary implementation ran at **81ms**. The PyTorch FP32 baseline was 11.18ms. I had made the model 7x slower by removing multiplications.

This is the kind of thing that makes you question your life choices.

## Training the Ternary Model

Before getting into the engine, a quick note on training. I used TWN (Ternary Weight Networks) ternarization with a threshold of delta = 0.7 * mean(|w|). Training ran for 200 epochs with SGD, cosine annealing, and a learning rate of 0.05. Final accuracy landed at 94.46%, down about 1% from the FP32 baseline's 95.54%. That is a reasonable trade for 12.5x compression.

## The Journey: 19 Experiments, Mostly Failures

The single most important thing I learned in this project is that you should never optimize without profiling first. I use Intel VTune extensively, and it saved me from going down dead ends more than once. Though, to be honest, it did not save me from all of them.

### Attempt 1: The Obvious Approach (FP32 Ternary)

My first attempt kept activations in FP32 and used ternary weights to skip multiplications. Just additions, subtractions, and zeros. This ran at 81ms. The reason is straightforward once you look at the hardware: an FP32 add processes 8 values per AVX2 instruction. An INT8 multiply-accumulate using `dpbusd` (AVX-VNNI) processes **32 values per instruction**. By avoiding multiplications, I was leaving 4x throughput on the table.

Lesson: do not outsmart the hardware. Use the instructions it gives you.

### Attempt 2: INT16 Add/Sub

Maybe if I dropped to INT16, additions would be faster? They were not. 17.5ms. Half the throughput of dpbusd because INT16 packs 16 values per register instead of 32. The multiply in dpbusd is essentially free since the weights are {-1, 0, +1} anyway, and you get double the throughput.

### Attempt 3: The dpbusd Breakthrough

This is where things got interesting. `_mm256_dpbusd_epi32` is Intel's dot product instruction for INT8. It takes 32 unsigned bytes and 32 signed bytes, multiplies them pairwise, and accumulates into 8 int32 lanes. Thirty-two MACs in a single instruction.

The trick is the packing. Ternary weights are stored as 2-bit I2_S format: +1 encodes as 0b01, 0 as 0b00, -1 as 0b11. Four weights per byte. At runtime, these unpack to INT8 {-1, 0, +1} for VNNI. All activations stay in uint8 between layers, so the entire pipeline is INT8-resident. Static quantization scales are calibrated offline.

But the naive implementation was still slow. The problem was **horizontal reduction**. After each dpbusd, I had 8 partial sums in a 256-bit register that needed to be reduced to a single value. That reduction was eating the gains.

The fix was spatial packing: instead of reducing across the register, I arranged the data so that each of the 8 lanes corresponds to a different spatial output position. Layout is 8 spatial x 4 reduction elements. No horizontal reduction needed. Each lane independently accumulates its own output.

This single change took the engine from **79ms to 9.7ms**. An 8x speedup from rearranging memory layout.

### Attempt 4: 4-Bit Packing

At this point I wondered if I could keep weights in 4-bit format at runtime to save bandwidth. VTune showed 32.91ms and 8.2x more instructions than the INT8 path. The unpacking overhead dominated. Killed it immediately.

### Attempt 5: Fused Implicit GEMM

Instead of im2col followed by GEMM, why not fuse them? Compute the im2col indices on the fly during the matrix multiply. This ran at 58ms. VTune showed the scattered memory access pattern was destroying the prefetcher. Traditional im2col is a memcpy into a contiguous buffer, which is exactly what the hardware wants.

### Attempt 6: Specialized im2col for 3x3 Stride-1

VTune showed im2col was 30% of total runtime. Most of the convolutions in ResNet-18 are 3x3 with stride 1. For this specific case, I replaced the generic per-element gather with per-row memcpy. Instead of bounds-checking every element, I copy entire rows of the input feature map.

im2col dropped from 30% to 7.4% of runtime.

### Attempt 7: INT8 Shortcut Convolutions

ResNet's skip connections had a problem. The 1x1 shortcut convolutions were running in FP32, which meant I needed quantize/dequantize round-trips at every residual block. By implementing INT8 shortcut convolutions with the same dpbusd path, I eliminated all FP32 GEMM except the first conv layer. This removed a significant bottleneck.

### Attempt 8: The MLAS Discovery

This is the part that surprised me the most. I wanted to understand why ONNX Runtime's INT8 was so fast, so I dug into its internals. What I found: **oneDNN has no JIT INT8 kernel for AVX-VNNI**. It falls back to `ref_int8`, which is a reference implementation and extremely slow.

ONNX Runtime's actual speed comes from **MLAS** (Microsoft Linear Algebra Subroutines), not oneDNN. I disassembled the MLAS GEMM kernel and found it uses register-only accumulators across ymm4 through ymm15, with a 6-OC x 16-N tiling strategy. No spills to memory during the inner loop.

This inspired my next optimization: a 14-OC unrolled GEMM kernel. By keeping more output channels in registers simultaneously, I cut GEMM time by 50%.

### Attempts 9-19: Threading and Final Tuning

The single-threaded engine was at 3.71ms. With OpenMP parallelism at 6 threads: **0.99ms**. That is 2.6x faster than ONNX INT8 Dynamic (2.54ms) and within 13% of ONNX INT8 Static (0.88ms) at the same thread count — with a 3.2x smaller model.

## Final Results

| Method | 1-Thread | 6-Thread | Accuracy | Model Size |
|--------|----------|----------|----------|------------|
| PyTorch FP32 | 11.18ms | 6.50ms | 95.54% | 44.69 MB |
| ONNX INT8 Dynamic | 4.38ms | 2.54ms | 95.48% | 11.23 MB |
| ONNX INT8 Static | 2.24ms | 0.88ms | 95.43% | 11.23 MB |
| C++ Ternary Engine | 3.71ms | 0.99ms | 94.46% | 3.56 MB |

All numbers at 6 threads, median over 200 runs. The ternary engine is 2.6x faster than ONNX INT8 Dynamic and within 13% of ONNX INT8 Static, while using a model 3.2x smaller. The accuracy cost is about 1% compared to FP32. Total workspace memory is 9216 KB, preallocated and reused per inference.

## What I Actually Learned

**Most "optimizations" make things worse.** Of 19 experiments, more than half resulted in slower performance. Fused implicit GEMM sounded brilliant and was 6x slower than the naive approach. INT16 add/sub seemed like a natural fit for ternary and was 17x slower than dpbusd. 4-bit packing should have saved memory bandwidth and cost 3x more instructions.

**Profile before you optimize.** Every breakthrough came from looking at VTune output and finding the actual bottleneck. Every failure came from reasoning about what should be fast without measuring what was.

**Read the disassembly.** I would not have found the MLAS tiling strategy without disassembling ONNX Runtime. I would not have understood the horizontal reduction bottleneck without looking at the generated code. The compiler does not always do what you think it does.

**Use the hardware's instructions, not your own cleverness.** dpbusd does a multiply and an accumulate. The multiply is wasted on ternary weights. I spent a long time trying to avoid that wasted multiply, and every alternative was slower because nothing else processes 32 elements per instruction.

**Memory layout is everything.** The single biggest speedup in the entire project, 79ms to 9.7ms, came from rearranging how data was packed into SIMD lanes. Not a different algorithm. Not a different instruction. Just a different memory layout.

---

If you are working on inference optimization, my advice is simple: open VTune, look at what is actually slow, and fix that. Resist the urge to optimize what you think should be slow. The hardware will surprise you every time.

---

*All benchmarks on Intel Core i5-13400F (Raptor Lake), AVX-VNNI, Ubuntu. Code is C++ with intrinsics.*
