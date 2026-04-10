"""Sweep thread counts for C++ Ternary Engine and ONNX INT8 Static."""
import subprocess, time, re, sys
import numpy as np

ONNX_MODEL = "export/resnet18_int8_static.onnx"
CPP_BIN = "inference/build/ternary_infer"
CPP_MODEL = "export/ternary_resnet18.bin"
CPP_SCALES = "export/static_scales.json"
WARMUP = 100
RUNS = 500

def bench_cpp(threads):
    env = {"OMP_NUM_THREADS": str(threads)}
    r = subprocess.run(
        [CPP_BIN, CPP_MODEL, CPP_SCALES],
        capture_output=True, text=True, env={**__import__('os').environ, **env}
    )
    m = re.search(r'Median:\s+([\d.]+)', r.stdout)
    return float(m.group(1)) if m else None

def bench_onnx(threads):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(ONNX_MODEL, opts, providers=['CPUExecutionProvider'])
    inp = np.random.randn(1, 3, 32, 32).astype(np.float32)
    name = sess.get_inputs()[0].name
    for _ in range(WARMUP):
        sess.run(None, {name: inp})
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        sess.run(None, {name: inp})
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))

thread_counts = [1, 2, 3, 4, 5, 6]

print(f"{'Threads':>7} | {'C++ Ternary':>12} | {'ONNX INT8 Static':>16} | {'Winner':>10} | {'Gap':>8}")
print("-" * 72)

for t in thread_counts:
    cpp = bench_cpp(t)
    onnx = bench_onnx(t)
    winner = "C++ Ternary" if cpp < onnx else "ONNX INT8"
    gap = abs(cpp - onnx) / min(cpp, onnx) * 100
    marker = "<--" if cpp < onnx else ""
    print(f"{t:>7} | {cpp:>10.3f}ms | {onnx:>14.3f}ms | {winner:>10} | {gap:>6.1f}% {marker}")
