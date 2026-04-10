"""
Verify whether ONNX Runtime INT8 Static uses NHWC layout and eliminates im2col.
Checks:
1. ONNX graph node types and attributes (any layout transforms?)
2. ORT profiling to see if im2col/reorder ops appear
3. ORT session graph optimizations and execution plan
"""
import os, sys, json, time
import numpy as np

def main():
    import onnxruntime as ort
    import onnx

    model_path = "export/resnet18_int8_static.onnx"
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found. Run benchmark_onnx_int8.py first.")
        sys.exit(1)

    # ── 1. Check ONNX graph for layout info ──
    print("=" * 60)
    print("1. ONNX GRAPH ANALYSIS")
    print("=" * 60)
    model = onnx.load(model_path)
    graph = model.graph

    op_counts = {}
    layout_ops = []
    for node in graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
        # Check for Transpose or layout-related ops
        if node.op_type in ("Transpose", "Reshape", "NhwcConv", "NhwcMaxPool"):
            layout_ops.append(node)
        # Check Conv nodes for layout attributes
        if "Conv" in node.op_type:
            for attr in node.attribute:
                if attr.name == "channels_last" or "nhwc" in attr.name.lower():
                    layout_ops.append(node)

    print("\nOp type counts:")
    for op, cnt in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"  {op:<30s} {cnt}")

    print(f"\nLayout-related ops found: {len(layout_ops)}")
    for op in layout_ops:
        print(f"  {op.op_type}: {op.name}")
        for attr in op.attribute:
            print(f"    {attr.name} = {attr}")

    # ── 2. Check ORT graph optimizations (save optimized model) ──
    print("\n" + "=" * 60)
    print("2. ORT OPTIMIZED GRAPH (after graph transforms)")
    print("=" * 60)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.optimized_model_filepath = "export/resnet18_int8_static_optimized.onnx"

    session = ort.InferenceSession(model_path, opts, providers=['CPUExecutionProvider'])

    # Load the optimized model and check for NHWC ops
    if os.path.exists(opts.optimized_model_filepath):
        opt_model = onnx.load(opts.optimized_model_filepath)
        opt_op_counts = {}
        nhwc_ops = []
        transpose_ops = []
        for node in opt_model.graph.node:
            opt_op_counts[node.op_type] = opt_op_counts.get(node.op_type, 0) + 1
            if "nhwc" in node.op_type.lower() or "Nhwc" in node.op_type:
                nhwc_ops.append(node)
            if node.op_type == "Transpose":
                transpose_ops.append(node)
            # Check for layout_transformer inserted nodes
            if "layout" in node.name.lower() or "nhwc" in node.name.lower():
                nhwc_ops.append(node)

        print("\nOptimized graph op counts:")
        for op, cnt in sorted(opt_op_counts.items(), key=lambda x: -x[1]):
            print(f"  {op:<30s} {cnt}")

        print(f"\nNHWC-specific ops: {len(nhwc_ops)}")
        for op in nhwc_ops:
            print(f"  {op.op_type}: {op.name}")

        print(f"\nTranspose ops: {len(transpose_ops)}")
        for op in transpose_ops:
            print(f"  {op.name}")

        # Check if any fused conv nodes exist
        fused_ops = [n for n in opt_model.graph.node if "fuse" in n.op_type.lower() or "Fused" in n.op_type]
        print(f"\nFused ops: {len(fused_ops)}")
        for op in fused_ops:
            print(f"  {op.op_type}: {op.name}")

        os.remove(opts.optimized_model_filepath)

    # ── 3. Profile execution to see actual kernel names ──
    print("\n" + "=" * 60)
    print("3. ORT EXECUTION PROFILE (kernel-level)")
    print("=" * 60)

    opts2 = ort.SessionOptions()
    opts2.intra_op_num_threads = 1
    opts2.inter_op_num_threads = 1
    opts2.enable_profiling = True

    session2 = ort.InferenceSession(model_path, opts2, providers=['CPUExecutionProvider'])
    input_name = session2.get_inputs()[0].name
    dummy = np.random.randn(1, 3, 32, 32).astype(np.float32)

    # Warmup
    for _ in range(10):
        session2.run(None, {input_name: dummy})

    # Profile run
    session2.run(None, {input_name: dummy})
    profile_file = session2.end_profiling()

    with open(profile_file, "r") as f:
        data = json.load(f)

    # Collect all node-level ops
    node_ops = []
    for entry in data:
        if entry.get("cat") == "Node":
            name = entry.get("name", "")
            dur = entry.get("dur", 0)
            # Get op_name from args if available
            args = entry.get("args", {})
            op_name = args.get("op_name", "")
            provider = args.get("provider", "")
            node_ops.append({
                "name": name,
                "op_name": op_name,
                "dur_us": dur,
                "provider": provider
            })

    # Sort by duration
    node_ops.sort(key=lambda x: -x["dur_us"])

    print("\nAll operators by time:")
    total_us = sum(op["dur_us"] for op in node_ops)
    for op in node_ops[:30]:
        pct = 100.0 * op["dur_us"] / total_us if total_us > 0 else 0
        print(f"  {op['name']:<50s} {op['op_name']:<15s} {op['dur_us']:>6d} us ({pct:5.1f}%)")

    # Check for im2col or reorder in names
    print(f"\nTotal node time: {total_us} us")

    im2col_ops = [op for op in node_ops if "im2col" in op["name"].lower() or "reorder" in op["name"].lower()]
    print(f"\nim2col/reorder ops found: {len(im2col_ops)}")
    for op in im2col_ops:
        print(f"  {op['name']} {op['dur_us']} us")

    nhwc_kernel_ops = [op for op in node_ops if "nhwc" in op["name"].lower()]
    print(f"\nNHWC kernel ops found: {len(nhwc_kernel_ops)}")
    for op in nhwc_kernel_ops:
        print(f"  {op['name']} {op['dur_us']} us")

    # Cleanup
    os.remove(profile_file)

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)


if __name__ == "__main__":
    main()
