# Optimization Journal

## Target
Beat or match ONNX INT8 Static (2.56ms 1T, 1.08ms 6T) on Intel i5-13400F (Raptor Lake, AVX2+VNNI).

---

## Experiment 0: Baseline Benchmarks

| Method | 1-Thread | Multi-T | Accuracy | Size |
|---|---|---|---|---|
| PyTorch FP32 | 11.18ms | 6.50ms | 95.54% | 44.69 MB |
| PyTorch Ternary | 109.14ms | 33.67ms | 94.47% | 44.69 MB |
| ONNX FP32 | 9.75ms | 2.49ms | 95.54% | 44.69 MB |
| ONNX INT8 Dynamic | 4.26ms | 2.74ms | 95.48% | 11.23 MB |
| ONNX INT8 Static | 2.56ms | 1.08ms | 95.43% | 11.23 MB |

---

## Experiment 1: Naive C++ Engine (FP32 AVX2)

**What**: FP32 im2col + GEMM with AVX2 FMA. BN fused into conv weights.

**Result**: 79.26ms 1T. Correct baseline (95% accuracy on 500 images).

**Takeaway**: 31× slower than ONNX INT8 Static.

---

## Experiment 2: Transposed dpbusd VNNI (no OC tiling)

**What**: Dynamic INT8 quantize → transpose to [col_cols][cr_pad] → dpbusd per (OC, spatial_pos) with horizontal reduction.

**Result**: 17ms 1T, 8.8ms 6T.

**VTune**: CPI=0.294, Port 6 at 54.6% — horizontal reduction shuffles saturating a single execution port.

**Takeaway**: dpbusd throughput wasted on horizontal sums. 6.6× slower than ONNX.

---

## Experiment 3: INT16 add/sub approaches [FAILED]

**What**: Tried branchless sign_epi16 (17.5ms), branching add/sub (50ms), sparse pos/neg lists (17.9ms).

**Why failed**: INT16 has half the throughput (16 vs 32 ops/instruction). Sparse index lists destroy cache locality. sign_epi16 goes through the same saturated port 6.

---

## Experiment 4: FP32 ternary add/sub + 8-OC tiling [FAILED]

**What**: Skip quantization entirely, work on FP32 activations with ternary add/sub.

**Result**: 81ms 1T. FP32 add_ps processes 8 values vs INT16's 16 — throughput loss exceeds quantization savings.

---

## Experiment 5: 8-OC tiling on INT16 [FAILED]

**What**: Process 8 OCs together, reuse activation rows.

**Result**: 68ms 1T. L1 cache thrashing — 8 INT16 accumulators × large spatial dims exceed L1.

---

## Experiment 6: 3-group accumulation (BitNet TL2 inspired)

**What**: Process 3 reduction rows per spatial loop iteration using sign_epi16.

**Result**: 20.8ms 1T, 8.1ms 6T. Better MT than dpbusd but worse 1T.

---

## Experiment 7: Spatial-packed dpbusd — BREAKTHROUGH

**What**: Restructured dpbusd so each lane processes a DIFFERENT spatial position (8 spatial × 4 reduction per instruction). Weight broadcast to all lanes. Eliminated horizontal reduction entirely.

**Result with 16-OC tiling**: 9.7ms 1T, 7.5ms 6T.

**VTune**: CPI improved 0.294→0.208, Retiring 54%→72%, Port 6 pressure reduced, Split Loads 6%→1%.

**Takeaway**: Eliminating horizontal reduction + OC tiling = 1.75× speedup.

---

## Experiment 8: Optimized quantize+repack

**What**: Vectorized quantize (AVX2 packus), 8×4 blocked transpose for repack. Eliminated intermediate buffer.

**Result**: 9.0ms 1T, 6.2ms 6T.

---

## Experiment 9: Preallocated workspace buffers

**What**: Moved col_u8 + col_packed from per-call std::vector (malloc/free 16× per inference) to Engine-level persistent buffers.

**Result**: 7.3ms 1T, 5.0ms 6T. Eliminated 4000 malloc/free cycles and memset overhead.

---

## Experiment 10: INT8-Resident static quantization

**What**: Keep activations in uint8 between layers. Static calibrated scales. im2col on bytes (4× less data). No FP32→INT8 conversion between ternary convs.

**Result**: 7.2ms 1T, 5.3ms 6T. Eliminated 24.5% FP32 im2col + 9.7% findmax overhead.

---

## Experiment 11: Fused residual add+ReLU+requant in GEMM epilogue

**What**: Conv2 output stays as int32 in registers. Add residual (uint8 or FP32), ReLU, requant to uint8 — all in the GEMM output loop. Zero intermediate FP32 tensors.

**Result**: 6.9ms 1T, 4.8ms 6T.

---

## Experiment 12: 3×3 stride-1 specialized im2col

**What**: Replace per-element bounds-check im2col with memcpy-per-row for 3×3 stride 1 pad 1 (12 of 16 ternary convs). Each im2col row is a shifted copy of input row.

**Result**: 5.4ms 1T, 3.3ms 6T. im2col dropped from 30% → 7.4%.

**Takeaway**: Biggest single-optimization win on the im2col path.

---

## Experiment 13: Fused conv1 ReLU+requant epilogue

**What**: Extended fused epilogue (mode 1) for conv1 in each block: dequant → ReLU → uint8 in GEMM output. Eliminated separate relu_inplace + quantize_to_u8 passes.

**Result**: 5.6ms 1T, 3.3ms 6T. Marginal — conv1 fuse is only 16 of 32 ternary calls.

---

## Experiment 14: Fused implicit GEMM [FAILED]

**What**: Eliminate im2col+repack entirely by gathering bytes directly from input inside the GEMM loop. Precomputed mapping tables.

**Result**: 58ms (full fusion), 10.5ms (single-buffer fusion). Scattered byte reads with index arithmetic are slower than sequential im2col + sequential repack.

**Takeaway**: Sequential memory access beats fewer passes with scattered access.

---

## Experiment 15: INT8 shortcut convs via dpbusd

**What**: Quantize FP32 shortcut weights to INT8 at load time (per-channel scaling). Route 1×1 stride-2 shortcuts through same dpbusd spatial-packed path as ternary convs. Eliminated all FP32 GEMM except first conv.

**Result**: 4.9ms 1T, 2.7ms 6T. Engine::forward overhead dropped from 20.8% → 3.0%.

---

## Experiment 16: Precomputed per-OC constants

**What**: Precompute `alpha[oc] * fused_scale[oc]` at model load time. Runtime just multiplies by act_scale (single scalar).

**Result**: Marginal improvement, included with Experiment 15.

---

## Experiment 17: SIMD-vectorized repack (SSE unpack interleave)

**What**: Replaced scalar 8×4 byte repack with SSE `unpacklo_epi8` + `unpacklo_epi16` 4×8 transpose. 4 loads + 3 shuffles + 2 stores per 32-byte block vs 32 scalar byte stores.

**Result**: 4.5ms 1T, 2.15ms 6T. Repack overhead dropped from 22.8% to ~16%.

---

## Final Results

| Method | 1-Thread | 6-Thread | Accuracy | Size |
|---|---|---|---|---|
| PyTorch FP32 | 11.18ms | 6.50ms | 95.54% | 44.69 MB |
| PyTorch Ternary | 109.14ms | 33.67ms | 94.47% | 44.69 MB |
| ONNX FP32 | 9.75ms | 2.49ms | 95.54% | 44.69 MB |
| ONNX INT8 Dynamic | 4.26ms | 2.74ms | 95.48% | 11.23 MB |
| ONNX INT8 Static | 2.56ms | 1.08ms | 95.43% | 11.23 MB |
| **C++ Ternary Engine** | **4.5ms** | **2.15ms** | **95.00%** | **3.56 MB** |

### Speedup vs baselines:
- vs PyTorch Ternary: **24.2× (1T)**, **15.7× (6T)**
- vs PyTorch FP32: **2.5× (1T)**, **3.0× (6T)**
- vs ONNX INT8 Dynamic: **within 6% (1T)**, **1.3× faster (6T)**
- vs ONNX INT8 Static: **1.8× slower (1T)**, **2.0× slower (6T)**
- Model size: **3.2× smaller than ONNX INT8**, **12.5× smaller than FP32**

### Final VTune profile (1T):
```
dpbusd GEMM:         69.7%   CPI=0.171, Ret=91.7%  (near-peak)
im2col + repack:     16.4%   CPI=0.311, Ret=51.6%
memcpy (im2col):      4.6%
Engine::forward:       4.4%   (first conv only)
memset:                1.5%
other:                 3.4%
```

### Remaining gap to ONNX INT8 Static (2×):
oneDNN uses fused implicit GEMM that avoids im2col/repack entirely (~21% of our time). Their register-tiled micro-kernels also achieve higher activation reuse through dual-axis tiling (OC × spatial simultaneously). Matching this would require building a full production-grade GEMM library.

### Key architectural insights:
1. **Ternary weights as INT8 with dpbusd** beats all ternary-specific tricks (add/sub, sign, sparsity, LUT) for CNN workloads with OC < 512.
2. **Spatial-packed dpbusd layout** (8 spatial × 4 reduction per instruction) eliminates horizontal reduction — the single biggest VTune-identified bottleneck.
3. **INT8-resident execution** with static scales eliminates ~35% of overhead (FP32 im2col, dynamic quantize, requantize).
4. **Sequential two-pass** (im2col → repack) beats single-pass scattered gather by 2× due to cache locality.
5. **3×3 stride-1 specialization** with memcpy-per-row cuts im2col from 30% → 7%.

---

## Experiment 18: 4-Bit QAT + sign_epi32 Add/Sub GEMM [FAILED]

**Hypothesis**: With 4-bit activations [0,15] and ternary weights {-1,0,+1}, every multiply is just +act, -act, or 0. Replace dpbusd with `_mm256_sign_epi32` + `_mm256_add_epi32`. Eliminates spatial-packed repack entirely (16.4% of time). Zero-weight skipping (~30% sparsity) provides further savings.

**Code changed**: New `engine_4bit.h` with sign+add GEMM, standard im2col (no repack). 4-bit QAT model trained with `train/train_ternary_4bit.py` (93.65% best accuracy at epoch 189).

**Before vs after**:
| Metric | INT8 dpbusd engine | 4-bit sign+add engine |
|---|---|---|
| Median (1T) | 4.70ms | 32.91ms |
| Accuracy | 94.46% | 93.56% |

**VTune evidence**:
| Function | 4-bit sign+add | INT8 dpbusd |
|---|---|---|
| GEMM kernel (insns) | 165.8B (CPI=0.215) | 20.3B (CPI=0.179) |
| GEMM wall time | 7.914s | 0.749s |
| im2col+repack | 0.100s (no repack) | 0.192s (with repack) |
| Retiring % | 70% | 68% |

**Why it failed**: 8.2× more instructions in the GEMM. sign+add processes 8 spatial × 1 reduction in ~3 instructions. dpbusd processes 8 spatial × 4 reduction in 1 instruction (32 MACs). Both pipelines are well-utilized (70% vs 68% retiring) — the gap is purely instruction count. Eliminating repack saved 0.092s but the GEMM regressed by 7.2s.

**Takeaway**: On VNNI hardware, dpbusd's 32 MACs/instruction with 1-cycle throughput is unbeatable. The hardware multiplier is "free" — ternary's trivial multiply doesn't help because the multiply wasn't the bottleneck.

---

## Experiment 19: Nibble-Packed Activations + dpbusd [FAILED]

**Hypothesis**: Store activations as 4-bit nibbles (TensorU4, 2 values/byte) between layers. Use direct-to-packed im2col (fused unpack + spatial-pack) to eliminate the separate repack step. dpbusd GEMM unchanged. Expected win: halved inter-layer memory + no repack pass.

**Code changed**: New `engine_4bit_dpbusd.h` with TensorU4 struct, SIMD nibble unpack, direct-to-packed im2col for 3×3s1p1 layers.

**Before vs after**:
| Metric | INT8 dpbusd engine | U4 nibble-packed dpbusd |
|---|---|---|
| Median (1T) | 4.70ms | 8.24ms |
| Accuracy | 94.46% | 93.64% |
| Activation storage | 1 byte/val | 0.5 byte/val |

**VTune evidence**:
| Function | U4 nibble-packed dpbusd | INT8 dpbusd |
|---|---|---|
| dpbusd GEMM (insns) | 21.8B | 20.3B |
| Outer func (unpack+im2col) | 27.9B insns | 2.7B insns |

**Why it failed**: The direct-to-packed im2col does element-by-element scalar nibble extraction with index arithmetic (c/kh/kw decomposition + bounds checks) for each of 32 values per block. This generated 27.9B instructions vs 2.7B for memcpy-based im2col + SIMD repack — a 10× instruction inflation.

CIFAR-10 activations (32–64KB) already fit in L1 cache. Halving them to 16–32KB provides zero cache benefit. The unpack + nibble extraction overhead far exceeds any bandwidth savings.

**Takeaway**: "Fewer bytes" doesn't mean "faster" when the data already fits in cache. The proven two-step approach (memcpy im2col + SIMD repack) is hard to beat because bulk memory operations are extremely efficient on modern CPUs. Fused single-pass approaches with scalar element access lose to sequential vectorized passes.
