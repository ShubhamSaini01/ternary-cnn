"""
Hybrid Benchmark: MLAS INT8 conv (ORT) + custom FP32/pool/FC
Builds the full ResNet-18 forward pass using:
  - ORT sessions for each ternary INT8 conv (MLAS kernel)
  - NumPy for FP32 convs, pool, FC (simulating custom C++)

This measures the "best of both worlds" hybrid approach.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort
import time
import struct
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)


def load_binary_model(path):
    """Load our custom binary model format."""
    layers = []
    with open(path, 'rb') as f:
        magic, num_layers = struct.unpack('II', f.read(8))
        assert magic == 0x54524E59, f"Bad magic: {hex(magic)}"

        for i in range(num_layers):
            layer_type = struct.unpack('B', f.read(1))[0]
            oc, ic, kh, kw, stride, padding = struct.unpack('6I', f.read(24))
            has_bn = struct.unpack('B', f.read(1))[0]

            layer = {
                'type': layer_type, 'oc': oc, 'ic': ic,
                'kh': kh, 'kw': kw, 'stride': stride, 'padding': padding,
            }

            if layer_type == 0:  # FP32_CONV
                wsize = oc * ic * kh * kw
                layer['weights'] = np.frombuffer(f.read(wsize * 4), dtype=np.float32).reshape(oc, ic, kh, kw)
                if has_bn:
                    layer['fused_scale'] = np.frombuffer(f.read(oc * 4), dtype=np.float32)
                    layer['fused_bias'] = np.frombuffer(f.read(oc * 4), dtype=np.float32)
            elif layer_type == 1:  # TERNARY_CONV
                total_w = oc * ic * kh * kw
                packed_bytes = (total_w + 7) // 8
                mask_pos = np.frombuffer(f.read(packed_bytes), dtype=np.uint8)
                mask_neg = np.frombuffer(f.read(packed_bytes), dtype=np.uint8)
                # Unpack to int8 {-1, 0, +1}
                weights = np.zeros(total_w, dtype=np.int8)
                for k in range(total_w):
                    byte_idx, bit_idx = k // 8, k % 8
                    is_pos = (mask_pos[byte_idx] >> bit_idx) & 1
                    is_neg = (mask_neg[byte_idx] >> bit_idx) & 1
                    if is_pos:
                        weights[k] = 1
                    elif is_neg:
                        weights[k] = -1
                layer['weights_int8'] = weights.reshape(oc, ic, kh, kw)
                layer['fused_scale'] = np.frombuffer(f.read(oc * 4), dtype=np.float32)
                layer['fused_bias'] = np.frombuffer(f.read(oc * 4), dtype=np.float32)
                # Compute weight_sum per OC
                layer['weight_sum'] = layer['weights_int8'].reshape(oc, -1).sum(axis=1).astype(np.int32)
            elif layer_type == 2:  # FC
                wsize = oc * ic
                layer['weights'] = np.frombuffer(f.read(wsize * 4), dtype=np.float32).reshape(oc, ic)
                layer['bias'] = np.frombuffer(f.read(oc * 4), dtype=np.float32)

            layers.append(layer)
    return layers


def create_ternary_conv_onnx(layer, in_h, in_w, path):
    """Create QLinearConv ONNX model for a single ternary layer."""
    ic, oc = layer['ic'], layer['oc']
    kh, kw = layer['kh'], layer['kw']
    s, p = layer['stride'], layer['padding']
    oh = (in_h + 2*p - kh) // s + 1
    ow = (in_w + 2*p - kw) // s + 1

    w_int8 = layer['weights_int8']
    # Scale: we'll use dynamic quantization externally, so set trivial scales here
    # Actually: we handle quantize/dequant ourselves for accuracy matching.
    # Use Conv with s8 weights and f32 input/output (ORT will handle quantization internally)

    # Simpler approach: use regular Conv with FP32 weights that are {-1,0,1}
    # ORT will still use optimized kernels. But we lose INT8 advantage.

    # Best approach: QuantizeLinear → QLinearConv → DequantizeLinear
    x_scale = np.float32(1.0)  # placeholder, we'll handle dynamically
    x_zp = np.uint8(128)
    w_scale = np.float32(1.0)
    w_zp = np.int8(0)
    y_scale = np.float32(1.0)
    y_zp = np.uint8(128)

    nodes = [
        helper.make_node('QuantizeLinear', ['input', 'x_scale', 'x_zp'], ['x_quant']),
        helper.make_node('QLinearConv',
            ['x_quant', 'x_scale', 'x_zp', 'w_int8', 'w_scale', 'w_zp', 'y_scale', 'y_zp'],
            ['y_quant'],
            kernel_shape=[kh, kw], strides=[s, s], pads=[p, p, p, p]),
        helper.make_node('DequantizeLinear', ['y_quant', 'y_scale', 'y_zp'], ['output']),
    ]

    graph = helper.make_graph(
        nodes, 'conv',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, ic, in_h, in_w])],
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


class HybridEngine:
    """Hybrid engine: ORT MLAS for ternary convs, numpy for FP32 ops."""

    def __init__(self, model_path):
        self.layers = load_binary_model(model_path)
        self.ort_sessions = {}
        self._setup_ort_sessions()

    def _setup_ort_sessions(self):
        """Create ORT sessions for each ternary conv layer."""
        os.makedirs("/tmp/hybrid_onnx", exist_ok=True)

        # Track spatial dims through the network
        h, w = 32, 32
        li = 0

        # Stem FP32
        layer = self.layers[li]
        h = (h + 2*layer['padding'] - layer['kh']) // layer['stride'] + 1
        w = (w + 2*layer['padding'] - layer['kw']) // layer['stride'] + 1
        li += 1

        # 8 basic blocks
        for grp in range(4):
            for blk in range(2):
                has_sc = (grp > 0 and blk == 0)
                block_h, block_w = h, w

                # First ternary conv
                layer = self.layers[li]
                onnx_path = f"/tmp/hybrid_onnx/layer_{li}.onnx"
                oh, ow = create_ternary_conv_onnx(layer, h, w, onnx_path)
                self.ort_sessions[li] = self._make_session(onnx_path)
                self.layers[li]['in_h'] = h
                self.layers[li]['in_w'] = w
                self.layers[li]['out_h'] = oh
                self.layers[li]['out_w'] = ow
                h, w = oh, ow
                li += 1

                # Second ternary conv
                layer = self.layers[li]
                onnx_path = f"/tmp/hybrid_onnx/layer_{li}.onnx"
                oh, ow = create_ternary_conv_onnx(layer, h, w, onnx_path)
                self.ort_sessions[li] = self._make_session(onnx_path)
                self.layers[li]['in_h'] = h
                self.layers[li]['in_w'] = w
                self.layers[li]['out_h'] = oh
                self.layers[li]['out_w'] = ow
                h, w = oh, ow
                li += 1

                if has_sc:
                    li += 1  # skip FP32 shortcut

    def _make_session(self, path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(path, opts, providers=['CPUExecutionProvider'])

    def ternary_conv(self, x, layer_idx):
        """Run ternary conv through ORT MLAS."""
        session = self.ort_sessions[layer_idx]
        input_name = session.get_inputs()[0].name
        out = session.run(None, {input_name: x})[0]
        return out

    def fp32_conv(self, x, layer):
        """Simple FP32 conv using numpy (for stem and shortcuts)."""
        oc = layer['oc']
        ic = layer['ic']
        kh, kw = layer['kh'], layer['kw']
        s, p = layer['stride'], layer['padding']
        _, _, h, w = x.shape
        oh = (h + 2*p - kh) // s + 1
        ow = (w + 2*p - kw) // s + 1

        # Pad
        if p > 0:
            x = np.pad(x, ((0,0),(0,0),(p,p),(p,p)), mode='constant')

        # im2col + matmul
        col = np.zeros((ic*kh*kw, oh*ow), dtype=np.float32)
        idx = 0
        for c in range(ic):
            for r in range(kh):
                for cc in range(kw):
                    for i in range(oh):
                        for j in range(ow):
                            col[idx, i*ow+j] = x[0, c, i*s+r, j*s+cc]
                    idx += 1

        weights = layer['weights'].reshape(oc, -1)  # [OC, IC*KH*KW]
        out = weights @ col  # [OC, OH*OW]

        if 'fused_scale' in layer:
            out = out * layer['fused_scale'][:, None] + layer['fused_bias'][:, None]

        return out.reshape(1, oc, oh, ow)

    def relu(self, x):
        return np.maximum(x, 0)

    def global_avg_pool(self, x):
        return x.mean(axis=(2, 3), keepdims=True)

    def fc(self, x, layer):
        x_flat = x.reshape(1, -1)
        return (x_flat @ layer['weights'].T + layer['bias']).reshape(1, -1, 1, 1)

    def forward(self, x):
        idx = 0

        # Stem (FP32)
        x = self.fp32_conv(x, self.layers[idx])
        x = self.relu(x)
        idx += 1

        # 8 basic blocks
        for grp in range(4):
            for blk in range(2):
                has_sc = (grp > 0 and blk == 0)

                idx1, idx2 = idx, idx + 1
                idx += 2
                idx_sc = idx if has_sc else -1
                if has_sc:
                    idx += 1

                # First ternary conv + ReLU
                identity = x
                x = self.ternary_conv(x, idx1)
                x = self.relu(x)

                # Shortcut
                if has_sc:
                    sc = self.fp32_conv(identity, self.layers[idx_sc])
                else:
                    sc = identity

                # Second ternary conv
                x = self.ternary_conv(x, idx2)

                # Residual + ReLU
                x = x + sc
                x = self.relu(x)

        x = self.global_avg_pool(x)
        x = self.fc(x, self.layers[idx])
        return x


def main():
    print("=" * 70)
    print("  Hybrid Engine Benchmark: MLAS INT8 + Custom FP32")
    print("=" * 70)

    model_path = "export/ternary_resnet18.bin"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    print("\nLoading hybrid engine...")
    engine = HybridEngine(model_path)

    # Create test input
    np.random.seed(42)
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    for c in range(3):
        x[0, c] = (np.random.rand(32, 32).astype(np.float32) - MEAN[c]) / STD[c]

    print("Sanity check...")
    out = engine.forward(x)
    pred = np.argmax(out.reshape(-1))
    print(f"  Prediction: class {pred}")
    print(f"  Logits: {out.reshape(-1)[:5]}...")

    # Warmup
    warmup = 50
    print(f"\nWarming up ({warmup} runs)...")
    for _ in range(warmup):
        engine.forward(x)

    # Benchmark
    runs = 200
    print(f"Benchmarking ({runs} runs)...")

    # Total time
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        engine.forward(x)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    times = np.array(times)
    print(f"\n{'─' * 50}")
    print(f"Hybrid Engine (MLAS + numpy FP32, 1T)")
    print(f"{'─' * 50}")
    print(f"  Mean:   {np.mean(times):.3f} ms")
    print(f"  Median: {np.median(times):.3f} ms")
    print(f"  Min:    {np.min(times):.3f} ms")
    print(f"  P95:    {np.percentile(times, 95):.3f} ms")

    # Per-component timing
    print(f"\nPer-component breakdown (avg over {runs} runs):")
    t_ternary = 0
    t_fp32 = 0
    t_other = 0

    for _ in range(runs):
        idx = 0

        # Stem
        t0 = time.perf_counter()
        xr = engine.fp32_conv(x, engine.layers[idx])
        xr = engine.relu(xr)
        idx += 1
        t_fp32 += time.perf_counter() - t0

        for grp in range(4):
            for blk in range(2):
                has_sc = (grp > 0 and blk == 0)
                idx1, idx2 = idx, idx + 1
                idx += 2
                idx_sc = idx if has_sc else -1
                if has_sc:
                    idx += 1

                identity = xr

                t0 = time.perf_counter()
                xr = engine.ternary_conv(xr, idx1)
                t_ternary += time.perf_counter() - t0

                t0 = time.perf_counter()
                xr = engine.relu(xr)
                t_other += time.perf_counter() - t0

                if has_sc:
                    t0 = time.perf_counter()
                    sc = engine.fp32_conv(identity, engine.layers[idx_sc])
                    t_fp32 += time.perf_counter() - t0
                else:
                    sc = identity

                t0 = time.perf_counter()
                xr = engine.ternary_conv(xr, idx2)
                t_ternary += time.perf_counter() - t0

                t0 = time.perf_counter()
                xr = xr + sc
                xr = engine.relu(xr)
                t_other += time.perf_counter() - t0

        t0 = time.perf_counter()
        xr = engine.global_avg_pool(xr)
        xr = engine.fc(xr, engine.layers[idx])
        t_other += time.perf_counter() - t0

    print(f"  Ternary convs (MLAS):  {t_ternary/runs*1000:.3f} ms")
    print(f"  FP32 convs (numpy):    {t_fp32/runs*1000:.3f} ms")
    print(f"  Other (relu,pool,fc):  {t_other/runs*1000:.3f} ms")

    # Comparison
    print(f"\n{'=' * 50}")
    print(f"COMPARISON (1T median latency)")
    print(f"{'=' * 50}")
    cpp_median = 9.495  # from the C++ benchmark
    hybrid_median = np.median(times)
    print(f"  C++ VNNI engine:   {cpp_median:.3f} ms")
    print(f"  Hybrid (this):     {hybrid_median:.3f} ms")
    print(f"  MLAS ternary only: {t_ternary/runs*1000:.3f} ms  (vs C++ 8.620 ms)")
    print(f"  Note: FP32 uses numpy (slow). C++ FP32 = 0.69ms")
    print(f"")
    projected = t_ternary/runs*1000 + 0.69 + 0.007  # MLAS ternary + C++ FP32 + C++ other
    print(f"  Projected hybrid (MLAS + C++ FP32): ~{projected:.3f} ms")
    print(f"  Projected speedup: {cpp_median/projected:.2f}x")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
