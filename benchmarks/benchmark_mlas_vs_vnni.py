"""
Per-layer comparison: MLAS (ONNX Runtime) INT8 conv vs hand-written VNNI.
Creates minimal single-conv ONNX models and benchmarks them to measure
MLAS's INT8 GEMM speed on each ResNet-18 layer shape.
"""
import numpy as np
import time
import os
import subprocess

def create_single_conv_onnx(ic, oc, h, w, k, s, p, path, static_quant=True):
    """Create a minimal ONNX model: single QuantizeLinear → QLinearConv → DequantizeLinear"""
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    oh = (h + 2*p - k) // s + 1
    ow = (w + 2*p - k) // s + 1

    # Create weights as INT8 ternary {-1, 0, 1}
    np.random.seed(42)
    w_fp32 = np.random.choice([-1, 0, 1], size=(oc, ic, k, k)).astype(np.float32)
    w_scale = np.float32(1.0 / 127.0)
    w_zp = np.int8(0)
    w_int8 = np.clip(np.round(w_fp32 / w_scale), -128, 127).astype(np.int8)

    # Activation quantization params
    x_scale = np.float32(1.0 / 127.0)
    x_zp = np.uint8(128)  # zero-point = 128 for u8

    # Output quantization params
    y_scale = np.float32(1.0 / 127.0)
    y_zp = np.uint8(128)

    # Build graph: input(f32) → QuantizeLinear → QLinearConv → DequantizeLinear → output(f32)
    nodes = []

    # QuantizeLinear: f32 → u8
    nodes.append(helper.make_node('QuantizeLinear', ['input', 'x_scale', 'x_zp'], ['x_quant']))

    # QLinearConv: u8 × s8 → u8
    nodes.append(helper.make_node('QLinearConv',
        ['x_quant', 'x_scale', 'x_zp',
         'w_int8', 'w_scale', 'w_zp',
         'y_scale', 'y_zp'],
        ['y_quant'],
        kernel_shape=[k, k],
        strides=[s, s],
        pads=[p, p, p, p],
    ))

    # DequantizeLinear: u8 → f32
    nodes.append(helper.make_node('DequantizeLinear', ['y_quant', 'y_scale', 'y_zp'], ['output']))

    graph = helper.make_graph(
        nodes,
        'single_conv',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, ic, h, w])],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, oc, oh, ow])],
        initializer=[
            numpy_helper.from_array(x_scale, 'x_scale'),
            numpy_helper.from_array(x_zp, 'x_zp'),
            numpy_helper.from_array(w_int8, 'w_int8'),
            numpy_helper.from_array(w_scale, 'w_scale'),
            numpy_helper.from_array(w_zp, 'w_zp'),
            numpy_helper.from_array(y_scale, 'y_scale'),
            numpy_helper.from_array(y_zp, 'y_zp'),
        ]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = 9
    onnx.save(model, path)
    return oh, ow


def benchmark_ort_conv(model_path, ic, h, w, warmup=200, runs=1000):
    """Benchmark a single-conv ONNX model through ORT (MLAS)"""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(model_path, opts, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, ic, h, w).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        session.run(None, {input_name: dummy})

    # Benchmark
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy})
        end = time.perf_counter()
        times.append((end - start) * 1e6)  # microseconds

    times = np.array(times)
    return np.median(times), np.mean(times), np.min(times), np.percentile(times, 95)


def main():
    print("=" * 70)
    print("  Per-Layer INT8 Conv Benchmark: MLAS (ONNX Runtime) vs Hand-Written VNNI")
    print("=" * 70)

    # ResNet-18 ternary conv layers (matching the C++ engine)
    layers = [
        # (name, ic, oc, h, w, k, stride, pad)
        ("L1:  64→64   32x32 s1", 64,  64,  32, 32, 3, 1, 1),
        ("L2:  64→64   32x32 s1", 64,  64,  32, 32, 3, 1, 1),
        ("L3:  64→64   32x32 s1", 64,  64,  32, 32, 3, 1, 1),
        ("L4:  64→64   32x32 s1", 64,  64,  32, 32, 3, 1, 1),
        ("L5:  64→128  32→16 s2", 64,  128, 32, 32, 3, 2, 1),
        ("L6:  128→128 16x16 s1", 128, 128, 16, 16, 3, 1, 1),
        ("L7:  128→128 16x16 s1", 128, 128, 16, 16, 3, 1, 1),
        ("L8:  128→128 16x16 s1", 128, 128, 16, 16, 3, 1, 1),
        ("L9:  128→256  16→8 s2", 128, 256, 16, 16, 3, 2, 1),
        ("L10: 256→256   8x8 s1", 256, 256,  8,  8, 3, 1, 1),
        ("L11: 256→256   8x8 s1", 256, 256,  8,  8, 3, 1, 1),
        ("L12: 256→256   8x8 s1", 256, 256,  8,  8, 3, 1, 1),
        ("L13: 256→512   8→4 s2", 256, 512,  8,  8, 3, 2, 1),
        ("L14: 512→512   4x4 s1", 512, 512,  4,  4, 3, 1, 1),
        ("L15: 512→512   4x4 s1", 512, 512,  4,  4, 3, 1, 1),
        ("L16: 512→512   4x4 s1", 512, 512,  4,  4, 3, 1, 1),
    ]

    os.makedirs("/tmp/ort_bench", exist_ok=True)

    print(f"\n{'Layer':<25s}  {'MLAS median':>12s}  {'MLAS mean':>12s}  {'MLAS min':>12s}  {'MLAS p95':>12s}")
    print("─" * 80)

    total_mlas = 0
    per_layer_times = []

    for name, ic, oc, h, w, k, s, p in layers:
        path = f"/tmp/ort_bench/conv_{ic}_{oc}_{h}_{s}.onnx"
        oh, ow = create_single_conv_onnx(ic, oc, h, w, k, s, p, path)

        median, mean, mn, p95 = benchmark_ort_conv(path, ic, h, w, warmup=200, runs=1000)
        total_mlas += median
        per_layer_times.append((name, median))

        print(f"{name:<25s}  {median:>9.1f} µs  {mean:>9.1f} µs  {mn:>9.1f} µs  {p95:>9.1f} µs")

    print("─" * 80)
    print(f"{'TOTAL (16 ternary convs)':<25s}  {total_mlas:>9.1f} µs")
    print(f"{'= ':<25s}  {total_mlas/1000:>9.3f} ms")

    # Now compare to C++ engine profile
    # The C++ engine reports per-inference ternary conv total
    # From the benchmark: total ternary = 8.62ms (quantize: 0.14, im2col: 0.85, compute: 7.63)
    print("\n" + "=" * 70)
    print("  COMPARISON")
    print("=" * 70)
    print(f"""
  MLAS total (16 ternary convs, per-inference):  {total_mlas/1000:.3f} ms
    (includes: quantize + conv + dequant, per ORT)

  Hand-written VNNI (from C++ profile, 1T):
    quantize:           0.140 ms
    im2col+interleave:  0.850 ms
    VNNI compute:       7.630 ms
    total ternary:      8.620 ms

  MLAS speedup on ternary convs: {8620/total_mlas:.2f}x

  Note: MLAS times include quantize+dequant overhead.
  The "pure compute" speedup would be higher since MLAS
  fuses more efficiently.
""")

    # Top 5 layers by time
    per_layer_times.sort(key=lambda x: -x[1])
    print("Top 5 layers by MLAS time:")
    for name, t in per_layer_times[:5]:
        print(f"  {name}: {t:.1f} µs ({t/total_mlas*100:.1f}%)")


if __name__ == "__main__":
    main()
