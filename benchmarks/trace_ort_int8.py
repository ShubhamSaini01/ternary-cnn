"""
Trace ONNX Runtime INT8 static inference — shows exactly what kernels MLAS uses.
Run with:
  python benchmarks/trace_ort_int8.py           # ORT profiling
  perf record -g python benchmarks/trace_ort_int8.py --perf   # then perf report
"""
import numpy as np
import onnxruntime as ort
import json
import sys
import os
import time

MODEL = "export/resnet18_int8_static.onnx"

def ort_profile_trace():
    """Use ORT's built-in profiler to get per-node kernel info."""
    print("=" * 70)
    print("  ORT Profiling: Per-Node Kernel Trace")
    print("=" * 70)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.enable_profiling = True
    # Enable verbose logging for kernel selection
    opts.log_severity_level = 0  # VERBOSE

    session = ort.InferenceSession(MODEL, opts, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 3, 32, 32).astype(np.float32)

    # Warmup
    for _ in range(20):
        session.run(None, {input_name: dummy})

    # Profiled run
    session.run(None, {input_name: dummy})
    profile_file = session.end_profiling()

    print(f"\nProfile saved to: {profile_file}")

    with open(profile_file) as f:
        data = json.load(f)

    # Extract node-level info
    print(f"\n{'Op Type':<25s} {'Node Name':<45s} {'Duration(us)':>12s} {'Provider':<15s}")
    print("─" * 100)

    nodes = []
    for entry in data:
        if entry.get("cat") == "Node":
            name = entry.get("name", "?")
            dur = entry.get("dur", 0)
            args = entry.get("args", {})
            op = args.get("op_name", "?")
            provider = args.get("provider", "?")
            thread = args.get("thread_scheduling_stats", "")
            nodes.append((op, name, dur, provider, args))

    # Sort by duration descending
    nodes.sort(key=lambda x: -x[2])

    total_us = 0
    for op, name, dur, provider, args in nodes:
        total_us += dur
        # Truncate long names
        short_name = name[:44] if len(name) > 44 else name
        print(f"{op:<25s} {short_name:<45s} {dur:>10d} us  {provider:<15s}")

    print("─" * 100)
    print(f"{'TOTAL':<25s} {'':45s} {total_us:>10d} us")

    # Group by op type
    print(f"\n\nGrouped by op type:")
    print(f"{'Op Type':<30s} {'Count':>6s} {'Total(us)':>12s} {'Avg(us)':>10s} {'%':>6s}")
    print("─" * 70)
    op_groups = {}
    for op, name, dur, provider, args in nodes:
        if op not in op_groups:
            op_groups[op] = {'count': 0, 'total': 0}
        op_groups[op]['count'] += 1
        op_groups[op]['total'] += dur

    for op, info in sorted(op_groups.items(), key=lambda x: -x[1]['total']):
        pct = 100 * info['total'] / total_us if total_us > 0 else 0
        avg = info['total'] / info['count']
        print(f"{op:<30s} {info['count']:>6d} {info['total']:>10d} us {avg:>10.1f} {pct:>5.1f}%")

    # Show full args for QLinearConv nodes (most interesting)
    print(f"\n\nQLinearConv node details:")
    print("─" * 70)
    for op, name, dur, provider, args in nodes:
        if 'QLinearConv' in op or 'Conv' in op:
            print(f"\n  {name} ({dur} us):")
            for k, v in sorted(args.items()):
                if k not in ('thread_scheduling_stats',):
                    print(f"    {k}: {v}")

    return profile_file


def perf_mode():
    """Run a tight loop for perf sampling."""
    print("Running tight inference loop for perf profiling...")
    print("(Use: perf record -g -p <pid> -- sleep 10)")
    print(f"PID: {os.getpid()}")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(MODEL, opts, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 3, 32, 32).astype(np.float32)

    # Warmup
    for _ in range(100):
        session.run(None, {input_name: dummy})

    # Tight loop for perf sampling
    print(f"Running 5000 iterations...")
    start = time.perf_counter()
    for _ in range(5000):
        session.run(None, {input_name: dummy})
    elapsed = time.perf_counter() - start
    print(f"Done. {elapsed:.2f}s, {elapsed/5000*1000:.3f} ms/iter")


def check_mlas_symbols():
    """List MLAS-related symbols in the ORT library."""
    import subprocess
    ort_lib = os.path.join(os.path.dirname(ort.__file__), "capi",
                           "onnxruntime_pybind11_state.cpython-310-x86_64-linux-gnu.so")
    if not os.path.exists(ort_lib):
        # Try the main lib
        ort_lib = os.path.join(os.path.dirname(ort.__file__), "capi", "libonnxruntime.so.1.23.2")

    print(f"\n{'=' * 70}")
    print(f"  MLAS Symbols in ORT ({os.path.basename(ort_lib)})")
    print(f"{'=' * 70}")

    try:
        result = subprocess.run(
            ['nm', '-DC', ort_lib],
            capture_output=True, text=True, timeout=30
        )
        symbols = result.stdout.split('\n')

        # Filter for MLAS/GEMM/Conv related symbols
        interesting = []
        for sym in symbols:
            lower = sym.lower()
            if any(kw in lower for kw in ['mlas', 'gemm', 'qlinear', 'conv_', 'quantize',
                                           'u8s8', 's8u8', 'vnni', 'avx', 'int8',
                                           'sbgemm', 'igemm', 'matmul']):
                interesting.append(sym.strip())

        # Deduplicate and sort
        interesting = sorted(set(interesting))

        print(f"\nFound {len(interesting)} relevant symbols:")
        for s in interesting[:80]:
            print(f"  {s}")
        if len(interesting) > 80:
            print(f"  ... ({len(interesting) - 80} more)")

    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    if "--perf" in sys.argv:
        perf_mode()
    elif "--symbols" in sys.argv:
        check_mlas_symbols()
    else:
        check_mlas_symbols()
        ort_profile_trace()
