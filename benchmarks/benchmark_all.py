"""
Unified benchmark: all methods, controlled thread counts, with accuracy.
Usage: python -u benchmarks/benchmark_all.py
"""

import sys, os, time
import numpy as np
import torch
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WARMUP = 50
RUNS = 200

def get_test_loader(bs=100):
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

def print_result(name, r, acc=None):
    acc_str = f"  acc={acc:.2f}%" if acc is not None else ""
    print(f"  {name:<35} median={r['median']:.3f}ms  mean={r['mean']:.3f}ms  min={r['min']:.3f}ms{acc_str}")

def pytorch_accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total

def onnx_accuracy(sess, loader):
    name = sess.get_inputs()[0].name
    correct = total = 0
    for images, labels in loader:
        outputs = sess.run(None, {name: images.numpy()})
        predicted = np.argmax(outputs[0], axis=1)
        total += labels.size(0)
        correct += (predicted == labels.numpy()).sum()
    return 100.0 * correct / total

def bench_pytorch(model_path, model_fn, label, threads, loader=None):
    torch.set_num_threads(threads)
    model = model_fn(num_classes=10)
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    dummy = torch.randn(1, 3, 32, 32)

    @torch.no_grad()
    def run(x):
        return model(x)

    r = bench_latency(run, dummy)
    actual_threads = torch.get_num_threads()

    acc = None
    if loader is not None and threads == 1:
        acc = pytorch_accuracy(model, loader)

    print_result(f"{label} ({actual_threads}T)", r, acc)
    return r, acc

def bench_onnx(model_path, label, threads, loader=None):
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
    thread_label = "all" if threads == 0 else str(threads)

    acc = None
    if loader is not None and threads == 1:
        acc = onnx_accuracy(sess, loader)

    print_result(f"{label} ({thread_label}T)", r, acc)
    return r, acc

def main():
    max_threads = torch.get_num_threads()
    print("=" * 70)
    print("UNIFIED BENCHMARK — All Methods, Controlled Thread Counts")
    print("=" * 70)
    print(f"CPU: {max_threads} threads available")
    print(f"Warmup: {WARMUP}, Benchmark runs: {RUNS}")
    print()

    print("Loading CIFAR-10 test set for accuracy verification...")
    loader = get_test_loader(bs=100)
    print()

    results = {}

    # ─── PyTorch FP32 ─────────────────────────────────────
    from models.resnet_fp import resnet18_cifar
    fp32_path = "checkpoints/resnet18_fp_best.pth"

    if os.path.exists(fp32_path):
        print("─── PyTorch FP32 Baseline ───")
        r1, acc = bench_pytorch(fp32_path, resnet18_cifar, "PyTorch FP32", threads=1, loader=loader)
        r2, _ = bench_pytorch(fp32_path, resnet18_cifar, "PyTorch FP32", threads=max_threads)
        results['pytorch_fp32'] = {'1t': r1, 'mt': r2, 'acc': acc, 'size': '44.69 MB'}
        print()

    # ─── PyTorch Ternary ──────────────────────────────────
    from models.resnet_ternary import ternary_resnet18_cifar
    tern_path = "checkpoints/resnet18_ternary_best.pth"

    if os.path.exists(tern_path):
        print("─── PyTorch Ternary (TWN) ───")
        r1, acc = bench_pytorch(tern_path, ternary_resnet18_cifar, "PyTorch Ternary", threads=1, loader=loader)
        r2, _ = bench_pytorch(tern_path, ternary_resnet18_cifar, "PyTorch Ternary", threads=max_threads)
        results['pytorch_tern'] = {'1t': r1, 'mt': r2, 'acc': acc, 'size': '44.69 MB'}
        print()

    # ─── ONNX FP32 ────────────────────────────────────────
    onnx_fp32 = "export/resnet18_fp32.onnx"
    if os.path.exists(onnx_fp32):
        print("─── ONNX FP32 ───")
        r1, acc = bench_onnx(onnx_fp32, "ONNX FP32", threads=1, loader=loader)
        r2, _ = bench_onnx(onnx_fp32, "ONNX FP32", threads=0)
        results['onnx_fp32'] = {'1t': r1, 'mt': r2, 'acc': acc, 'size': '44.69 MB'}
        print()

    # ─── ONNX INT8 Dynamic ────────────────────────────────
    onnx_int8d = "export/resnet18_int8_dynamic.onnx"
    if os.path.exists(onnx_int8d):
        print("─── ONNX INT8 Dynamic ───")
        r1, acc = bench_onnx(onnx_int8d, "ONNX INT8 Dynamic", threads=1, loader=loader)
        r2, _ = bench_onnx(onnx_int8d, "ONNX INT8 Dynamic", threads=0, loader=loader)
        results['onnx_int8d'] = {'1t': r1, 'mt': r2, 'acc': acc, 'size': '11.23 MB'}
        print()

    # ─── ONNX INT8 Static ────────────────────────────────
    onnx_int8s = "export/resnet18_int8_static.onnx"
    if os.path.exists(onnx_int8s):
        print("─── ONNX INT8 Static ───")
        r1, acc = bench_onnx(onnx_int8s, "ONNX INT8 Static", threads=1, loader=loader)
        r2, _ = bench_onnx(onnx_int8s, "ONNX INT8 Static", threads=0, loader=loader)
        results['onnx_int8s'] = {'1t': r1, 'mt': r2, 'acc': acc, 'size': '11.23 MB'}
        print()

    # ─── Our C++ Ternary Engine ─────────────────────────────
    cpp_bin = "inference/build/ternary_inference"
    model_bin = "export/ternary_resnet18.bin"
    test_bin = "inference/build/export/cifar10_test.bin"

    if os.path.exists(cpp_bin) and os.path.exists(model_bin):
        print("─── Our C++ Ternary Engine ───")
        import subprocess, re

        # Benchmark
        result = subprocess.run(
            [cpp_bin, model_bin, str(WARMUP), str(RUNS)],
            capture_output=True, text=True, timeout=300
        )

        cpp_median = None
        for line in result.stdout.splitlines():
            m = re.match(r'\s+Median:\s+([\d.]+)\s+ms', line)
            if m:
                cpp_median = float(m.group(1))

        # Accuracy
        cpp_acc = None
        if os.path.exists(test_bin):
            print("  Running accuracy on 10K test images...")
            acc_result = subprocess.run(
                [cpp_bin, model_bin, "--accuracy", test_bin],
                capture_output=True, text=True, timeout=600
            )
            m = re.search(r'Overall:.*?=\s+([\d.]+)%', acc_result.stdout)
            if m:
                cpp_acc = float(m.group(1))
        else:
            print(f"  Accuracy skipped — missing {test_bin}")
            print(f"  Run: python exports/export_test_data.py")

        if cpp_median is not None:
            acc_str = f"  acc={cpp_acc:.2f}%" if cpp_acc else ""
            print(f"  C++ Ternary V9b (1T)                median={cpp_median:.3f}ms{acc_str}")
        else:
            print("  C++ engine: failed to parse output")
            print("  First 300 chars:", result.stdout[:300])

        results['cpp'] = {'median': cpp_median, 'acc': cpp_acc, 'size': '3.50 MB'}
        print()
    else:
        print("─── Our C++ Ternary Engine ───")
        missing = []
        if not os.path.exists(cpp_bin): missing.append(cpp_bin)
        if not os.path.exists(model_bin): missing.append(model_bin)
        print(f"  Skipped — missing: {', '.join(missing)}")
        print()

    # ─── Summary Table ────────────────────────────────────
    print("=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"  {'Method':<25} {'1-Thread':>10} {'Multi-T':>10} {'Accuracy':>10} {'Size':>10}")
    print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

    order = [
        ("PyTorch FP32", 'pytorch_fp32'),
        ("PyTorch Ternary", 'pytorch_tern'),
        ("ONNX FP32", 'onnx_fp32'),
        ("ONNX INT8 Dynamic", 'onnx_int8d'),
        ("ONNX INT8 Static", 'onnx_int8s'),
        ("C++ V9b", 'cpp'),
    ]

    for name, key in order:
        if key not in results:
            continue
        r = results[key]
        size = r.get('size', '—')
        acc = f"{r['acc']:.2f}%" if r.get('acc') else "—"

        if key == 'cpp':
            t1 = f"{r['median']:.2f}ms" if r.get('median') else "—"
            mt = "TBD"
        else:
            t1 = f"{r['1t']['median']:.2f}ms" if r.get('1t') else "—"
            mt = f"{r['mt']['median']:.2f}ms" if r.get('mt') else "—"

        print(f"  {name:<25} {t1:>10} {mt:>10} {acc:>10} {size:>10}")

    print("=" * 70)


if __name__ == "__main__":
    main()