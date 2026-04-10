# Ternary ResNet-18 Inference Engine

A custom C++ ternary CNN inference engine for CIFAR-10, targeting Intel AVX-VNNI (Raptor Lake and newer). Achieves **12.5x model compression** vs FP32 and **3.2x smaller** than ONNX INT8.

**Author:** Shubham Saini

## Quick Start

```bash
# Build the inference engine
cd inference
make

# Export model weights (requires trained model)
python export/calibrate_and_export.py

# Run inference (50 warmup, 200 measurement runs, 6 threads)
export OMP_NUM_THREADS=6
./inference/build/ternary_infer export/ternary_resnet18.bin export/static_scales.json export/cifar10_test.bin 50 200
```

## Results

Benchmarked on Intel i5-13400F (Raptor Lake), AVX2 + AVX-VNNI, Linux x86_64. Median latency, 200 runs after 50 warmup.

| Method | 1-Thread | 6-Thread | Accuracy | Model Size |
|--------|----------|----------|----------|------------|
| PyTorch FP32 | 11.27 ms | 5.11 ms | 95.54% | 44.69 MB |
| ONNX INT8 Dynamic | 4.38 ms | 2.54 ms | 95.48% | 11.23 MB |
| ONNX INT8 Static | 2.24 ms | 0.88 ms | 95.43% | 11.23 MB |
| **C++ Ternary Engine** | **3.71 ms** | **0.99 ms** | **94.46%** | **3.56 MB** |

## Project Structure

```
train/              # PyTorch training scripts
models/             # Model definitions (ResNet-18 FP32 + ternary)
export/             # Binary weight export and calibration
inference/          # C++ inference engine (AVX-VNNI)
benchmarks/         # Latency, accuracy, and memory benchmarks
docs/               # Optimization journal, presentations
```

## Technical Details

- **Ternary weights:** 2-bit I2_S packing (4 weights per byte)
- **VNNI acceleration:** `_mm256_dpbusd_epi32` spatial-packed GEMM
- **INT8-resident pipeline:** Activations stay in INT8 between layers (no FP32 intermediates)
- **Optimized im2col:** Specialized 3x3 stride-1 implementation with memcpy-per-row
- **Parallelism:** OpenMP over output channels and spatial tiles

## Build Requirements

- C++17 compiler with AVX-VNNI support
- OpenMP
- Python 3.10+ with PyTorch (for training and export)
- ONNX Runtime (for comparison benchmarks)

Compiler flags: `-std=c++17 -O3 -march=native -mavx2 -mavxvnni -fopenmp`

## Build Commands

```bash
cd inference
make                # Builds main ternary engine
make 4bit           # 4-bit variant (experimental)
make 4bit-dpbusd    # 4-bit nibble-packed dpbusd variant
```

Output binaries are placed in `inference/build/`.

## How to Run

### Training

```bash
pip install -r requirements.txt
python train/train_ternary.py      # Train ternary ResNet-18
python train/train_fp32.py         # Train FP32 baseline
```

### Export

```bash
python export/calibrate_and_export.py  # Export weights and calibrate INT8 scales
```

### Inference

```bash
# Usage: ternary_infer <model.bin> <scales.json> [test_data] [warmup] [runs]
./inference/build/ternary_infer export/ternary_resnet18.bin export/static_scales.json export/cifar10_test.bin 50 200
```

### Controlling Thread Count

```bash
export OMP_NUM_THREADS=6
./inference/build/ternary_infer export/ternary_resnet18.bin export/static_scales.json export/cifar10_test.bin 50 200
```

### Benchmarks

```bash
# Full engine comparison (all 3 C++ engine variants)
./benchmarks/benchmark_all_engines.sh [threads] [warmup] [runs]
# Example: ./benchmarks/benchmark_all_engines.sh 6 50 200

# Unified benchmark (PyTorch + ONNX + C++ comparison)
python -u benchmarks/benchmark_all.py

# Thread scaling sweep
python benchmarks/thread_sweep.py

# Compare vs ONNX Runtime
python benchmarks/benchmark_vs_onnx.py
```

## VTune Microarchitecture Profile

Every optimization was driven by Intel VTune profiling. Key progression:

| Optimization | Latency (1T) | CPI | Retiring | What VTune Showed |
|-------------|-------------|-----|----------|-------------------|
| Naive dpbusd | 79 ms | 0.294 | 54% | Port 6 saturated — horizontal reduction shuffles |
| Spatial packing | 9.7 ms | 0.208 | 72% | Split loads 6% → 1%, port pressure balanced |
| + im2col specialized | 5.4 ms | — | — | im2col was 30% of time, per-element bounds checks |
| + INT8 shortcuts | 4.9 ms | — | — | FP32 shortcut convs were 20% of forward pass |
| + 14-OC unrolled GEMM | 3.96 ms | 0.179 | 82% | MLAS disassembly: register-only accumulators |
| **Final (1T)** | **3.71 ms** | **0.248** | **71.4%** | GEMM near-peak, im2col is remaining bottleneck |

**Final VTune state (1T):**
- GEMM kernel: CPI = 0.179, Retiring = 82%, 256-bit VNNI uOps = 25.8%
- Memory Bound: 7.1% (activations fit in L1/L2 at CIFAR-10 scale)
- Backend Bound: 23.2%, Front-End Bound: 12.4%

**6T VTune state:**
- CPI = 0.440, Retiring = 53.4%, Memory Bound = 22.0%
- OMP barriers: 23% of CPU time (libgomp spin-wait, ~57 syncs/inference)

## Documentation

- **[Inference Pipeline](docs/inference_pipeline.md)** -- Data flow diagram, dpbusd kernel, memory layout, threading model ([visual HTML version](docs/inference_pipeline.html))
- `docs/optimization_journal.md` -- 19 experiments with before/after measurements
- `docs/presentation.html` -- Slide deck
- `docs/engine_architecture.html` -- Engine data flow and design decisions
