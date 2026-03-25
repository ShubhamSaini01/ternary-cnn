"""
Unified benchmark: all methods, controlled thread counts.
Usage: python -u benchmarks/benchmark_all.py

Tests everything at 1 thread AND multi-thread for fair comparison.
"""

import sys, os, time
import numpy as np
import torch
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WARMUP = 50
RUNS = 200

def get_test_loader(bs=1):
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
    ])
    ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=t)
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)

def bench_latency(fn, input_data):
    for _ in range(WARMUP):
        fn(input_data)
    times = []
    for _ in range(RUNS):
        s = time.perf_counter()
        fn(input_data)
        e = time.perf_counter()
        times.append((e - s) * 1000)
    times = np.array(times)
    return {
        'mean': np.mean(times),
        'median': np.median(times),
        'min': np.min(times),
        'p95': np.percentile(times, 95),
    }

def print_result(name, r):
    print(f"  {name:<35} median={r['median']:.3f}ms  mean={r['mean']:.3f}ms  min={r['min']:.3f}ms  p95={r['p95']:.3f}ms")

def bench_pytorch(model_path, model_fn, label, threads):
    torch.set_num_threads(threads)
    model = model_fn(num_classes=10)
    ckpt = torch.load(model_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    dummy = torch.randn(1, 3, 32, 32)

    @torch.no_grad()
    def run(x):
        return model(x)

    r = bench_latency(run, dummy)
    actual_threads = torch.get_num_threads()
    print_result(f"{label} ({actual_threads}T)", r)
    return r

def bench_onnx(model_path, label, threads):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = threads
    sess = ort.InferenceSession(model_path, opts, providers=['CPUExecutionProvider'])
    name = sess.get_inputs()[0].name
    dummy = np.random.randn(1, 3, 32, 32).astype(np.float32)

    def run(x):
        return sess.run(None, {name: x})

    r = bench_latency(run, dummy)
    print_result(f"{label} ({threads}T)", r)
    return r

def main():
    max_threads = torch.get_num_threads()  # save BEFORE setting to 1
    print("=" * 70)
    print("UNIFIED BENCHMARK — All Methods, Controlled Thread Counts")
    print("=" * 70)
    print(f"CPU: {max_threads} threads available")
    print(f"Warmup: {WARMUP}, Benchmark runs: {RUNS}")
    print()

    # ─── PyTorch FP32 ─────────────────────────────────────
    from models.resnet_fp import resnet18_cifar
    fp32_path = "checkpoints/resnet18_fp_best.pth"

    if os.path.exists(fp32_path):
        print("─── PyTorch FP32 Baseline ───")
        bench_pytorch(fp32_path, resnet18_cifar, "PyTorch FP32", threads=1)
        bench_pytorch(fp32_path, resnet18_cifar, "PyTorch FP32", threads=max_threads)
        print()

    # ─── PyTorch Ternary ──────────────────────────────────
    from models.resnet_ternary import ternary_resnet18_cifar
    tern_path = "checkpoints/resnet18_ternary_best.pth"

    if os.path.exists(tern_path):
        print("─── PyTorch Ternary (TWN) ───")
        bench_pytorch(tern_path, ternary_resnet18_cifar, "PyTorch Ternary", threads=1)
        bench_pytorch(tern_path, ternary_resnet18_cifar, "PyTorch Ternary", threads=max_threads)
        print()

    # ─── ONNX FP32 ────────────────────────────────────────
    onnx_fp32 = "export/resnet18_fp32.onnx"
    if os.path.exists(onnx_fp32):
        print("─── ONNX FP32 ───")
        bench_onnx(onnx_fp32, "ONNX FP32", threads=1)
        bench_onnx(onnx_fp32, "ONNX FP32", threads=0)  # 0 = all threads
        print()

    # ─── ONNX INT8 Dynamic ────────────────────────────────
    onnx_int8d = "export/resnet18_int8_dynamic.onnx"
    if os.path.exists(onnx_int8d):
        print("─── ONNX INT8 Dynamic ───")
        bench_onnx(onnx_int8d, "ONNX INT8 Dynamic", threads=1)
        bench_onnx(onnx_int8d, "ONNX INT8 Dynamic", threads=0)
        print()

    # ─── ONNX INT8 Static ────────────────────────────────
    onnx_int8s = "export/resnet18_int8_static.onnx"
    if os.path.exists(onnx_int8s):
        print("─── ONNX INT8 Static ───")
        bench_onnx(onnx_int8s, "ONNX INT8 Static", threads=1)
        bench_onnx(onnx_int8s, "ONNX INT8 Static", threads=0)
        print()

    # ─── Summary ──────────────────────────────────────────
    print("=" * 70)
    print("NOTE: Our C++ ternary engine (V9) benchmarked separately:")
    print("  Single-threaded: 12.6ms median")
    print("  Multi-threaded:  TBD (add OpenMP)")
    print()
    print("Model sizes:")
    for path, label in [
        (fp32_path, "PyTorch FP32 checkpoint"),
        ("export/ternary_resnet18.bin", "Ternary packed"),
        (onnx_fp32, "ONNX FP32"),
        (onnx_int8d, "ONNX INT8 Dynamic"),
        (onnx_int8s, "ONNX INT8 Static"),
    ]:
        if os.path.exists(path):
            sz = os.path.getsize(path) / 1e6
            print(f"  {label:<30} {sz:.2f} MB")
    print("=" * 70)


if __name__ == "__main__":
    main()
