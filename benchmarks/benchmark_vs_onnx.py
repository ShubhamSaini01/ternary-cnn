"""
Head-to-head: all C++ engines vs ONNX INT8 Static (oneDNN backend).

ONNX Runtime's INT8 Static quantization uses oneDNN (MKL-DNN) for its
conv kernels — so this IS the oneDNN benchmark.

Usage:
    python benchmarks/benchmark_vs_onnx.py [--threads 1] [--runs 200]
"""

import os
import sys
import time
import argparse
import subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WARMUP = 50

def benchmark_onnx(model_path, label, num_threads, runs):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    session = ort.InferenceSession(model_path, opts, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 3, 32, 32).astype(np.float32)

    # Warmup
    for _ in range(WARMUP):
        session.run(None, {input_name: dummy})

    # Benchmark
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy})
        end = time.perf_counter()
        times.append((end - start) * 1000)

    times = np.array(sorted(times))
    return {
        "median": np.median(times),
        "mean": np.mean(times),
        "min": np.min(times),
        "p95": np.percentile(times, 95),
    }


def benchmark_onnx_accuracy(model_path, num_threads):
    import onnxruntime as ort
    from torchvision import datasets, transforms
    import torch

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(model_path, opts, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = datasets.CIFAR10(root=os.path.join(ROOT, 'data'), train=False, download=False, transform=transform)
    loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

    correct = total = 0
    for inputs, targets in loader:
        outputs = session.run(None, {input_name: inputs.numpy()})
        pred = np.argmax(outputs[0], axis=1)
        correct += (pred == targets.numpy()).sum()
        total += targets.size(0)

    return 100.0 * correct / total


def benchmark_cpp_engine(binary, model, scales, test_data, num_threads, warmup, runs):
    """Run a C++ engine binary and parse its output."""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(num_threads)

    cmd = [binary, model, scales, test_data, str(warmup), str(runs)]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    output = result.stdout

    # Parse
    acc = med = p95 = min_t = None
    for line in output.split("\n"):
        if "Overall:" in line:
            acc = float(line.split("=")[-1].strip().replace("%", ""))
        elif "Median:" in line:
            med = float(line.split()[-2])
        elif "P95:" in line:
            p95 = float(line.split()[-2])
        elif "Min:" in line:
            min_t = float(line.split()[-2])

    return {"accuracy": acc, "median": med, "p95": p95, "min": min_t}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()

    T = args.threads
    R = args.runs

    print("=" * 78)
    print(f"  All Engines vs ONNX INT8 Static (oneDNN)  |  Threads={T}  Runs={R}")
    print("=" * 78)
    print()

    # ── Paths ──
    onnx_static = os.path.join(ROOT, "export", "resnet18_int8_static.onnx")
    onnx_fp32   = os.path.join(ROOT, "export", "resnet18_fp32.onnx")
    cpp_int8    = os.path.join(ROOT, "inference", "build", "ternary_infer")
    cpp_4bit    = os.path.join(ROOT, "inference", "build", "ternary_infer_4bit")
    cpp_4bit_dp = os.path.join(ROOT, "inference", "build", "ternary_infer_4bit_dpbusd")
    model_int8  = os.path.join(ROOT, "export", "ternary_resnet18.bin")
    scales_int8 = os.path.join(ROOT, "export", "static_scales.json")
    model_4bit  = os.path.join(ROOT, "export", "ternary_resnet18_4bit.bin")
    scales_4bit = os.path.join(ROOT, "export", "static_scales_4bit.json")
    test_data   = os.path.join(ROOT, "export", "cifar10_test.bin")

    results = []

    # ── ONNX INT8 Static (oneDNN) ──
    if os.path.exists(onnx_static):
        print("[1/5] ONNX INT8 Static (oneDNN backend)...")
        r = benchmark_onnx(onnx_static, "ONNX INT8 Static", T, R)
        acc = benchmark_onnx_accuracy(onnx_static, T)
        r["accuracy"] = acc
        r["label"] = "ONNX INT8 Static (oneDNN)"
        r["size_mb"] = os.path.getsize(onnx_static) / 1e6
        results.append(r)
        print(f"       Median={r['median']:.3f}ms  Acc={acc:.2f}%")
    else:
        print("[1/5] ONNX INT8 Static — SKIPPED (file missing)")

    # ── ONNX FP32 ──
    if os.path.exists(onnx_fp32):
        print("[2/5] ONNX FP32...")
        r = benchmark_onnx(onnx_fp32, "ONNX FP32", T, R)
        r["accuracy"] = 95.54  # known from prior runs
        r["label"] = "ONNX FP32"
        r["size_mb"] = os.path.getsize(onnx_fp32) / 1e6
        results.append(r)
        print(f"       Median={r['median']:.3f}ms")
    else:
        print("[2/5] ONNX FP32 — SKIPPED (file missing)")

    # ── C++ INT8 dpbusd ──
    if os.path.exists(cpp_int8):
        print("[3/5] C++ INT8 dpbusd (our engine)...")
        r = benchmark_cpp_engine(cpp_int8, model_int8, scales_int8, test_data, T, WARMUP, R)
        r["label"] = "C++ INT8 dpbusd (ours)"
        r["size_mb"] = os.path.getsize(model_int8) / 1e6
        results.append(r)
        print(f"       Median={r['median']:.3f}ms  Acc={r['accuracy']:.2f}%")
    else:
        print("[3/5] C++ INT8 dpbusd — SKIPPED (binary missing)")

    # ── C++ 4-bit sign+add ──
    if os.path.exists(cpp_4bit):
        print("[4/5] C++ 4-bit sign+add...")
        r = benchmark_cpp_engine(cpp_4bit, model_4bit, scales_4bit, test_data, T, WARMUP, R)
        r["label"] = "C++ 4-bit sign+add"
        r["size_mb"] = os.path.getsize(model_4bit) / 1e6
        results.append(r)
        print(f"       Median={r['median']:.3f}ms  Acc={r['accuracy']:.2f}%")
    else:
        print("[4/5] C++ 4-bit sign+add — SKIPPED (binary missing)")

    # ── C++ 4-bit nibble-packed dpbusd ──
    if os.path.exists(cpp_4bit_dp):
        print("[5/5] C++ U4 nibble-packed dpbusd...")
        r = benchmark_cpp_engine(cpp_4bit_dp, model_4bit, scales_4bit, test_data, T, WARMUP, R)
        r["label"] = "C++ U4 nibble dpbusd"
        r["size_mb"] = os.path.getsize(model_4bit) / 1e6
        results.append(r)
        print(f"       Median={r['median']:.3f}ms  Acc={r['accuracy']:.2f}%")
    else:
        print("[5/5] C++ U4 nibble dpbusd — SKIPPED (binary missing)")

    # ── Summary Table ──
    print()
    print("=" * 78)
    print(f"{'Engine':<32} {'Accuracy':>8} {'Median':>10} {'P95':>10} {'Min':>10} {'Size':>8}")
    print("=" * 78)
    for r in results:
        acc_str = f"{r['accuracy']:.2f}%" if r.get('accuracy') else "—"
        med_str = f"{r['median']:.3f}ms" if r.get('median') else "—"
        p95_str = f"{r['p95']:.3f}ms" if r.get('p95') else "—"
        min_str = f"{r['min']:.3f}ms" if r.get('min') else "—"
        size_str = f"{r['size_mb']:.1f}MB" if r.get('size_mb') else "—"
        print(f"{r['label']:<32} {acc_str:>8} {med_str:>10} {p95_str:>10} {min_str:>10} {size_str:>8}")
    print("=" * 78)

    # ── Ratios ──
    if len(results) >= 3:
        onnx_med = results[0]["median"]
        ours_med = results[2]["median"]  # C++ INT8 dpbusd
        print(f"\n  ONNX INT8 Static vs our INT8 dpbusd:  {ours_med/onnx_med:.2f}x gap")
        print(f"  (ONNX uses oneDNN fused micro-kernels — this IS the oneDNN benchmark)")

    print(f"\n  OMP_NUM_THREADS={T}  |  Warmup={WARMUP}  |  Runs={R}")


if __name__ == "__main__":
    main()
