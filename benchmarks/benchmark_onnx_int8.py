"""
ONNX INT8 Quantization + Benchmark for comparison.
Usage:
    pip install onnx onnxruntime onnxruntime-extensions
    python benchmarks/benchmark_onnx_int8.py

Steps:
1. Export FP32 baseline to ONNX
2. Quantize to INT8 (dynamic quantization)
3. Also try static quantization with calibration data
4. Benchmark all variants on CPU
5. Compare against our PyTorch FP32 and Ternary numbers
"""

import sys
import os
import time
import numpy as np
import torch
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.resnet_fp import resnet18_cifar

# ─── Config ───────────────────────────────────────────────────
FP32_CHECKPOINT = "checkpoints/resnet18_fp_best.pth"
ONNX_FP32_PATH = "export/resnet18_fp32.onnx"
ONNX_INT8_DYN_PATH = "export/resnet18_int8_dynamic.onnx"
ONNX_INT8_STATIC_PATH = "export/resnet18_int8_static.onnx"
WARMUP_RUNS = 50
BENCHMARK_RUNS = 200
# ──────────────────────────────────────────────────────────────


def get_test_loader(batch_size=1):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    testset = datasets.CIFAR10(root='./data', train=False,
                               download=True, transform=transform_test)
    return torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=0)


# ─── Step 1: Export to ONNX ───────────────────────────────────
def export_to_onnx():
    print("Step 1: Exporting FP32 model to ONNX...")
    model = resnet18_cifar(num_classes=10)
    ckpt = torch.load(FP32_CHECKPOINT, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    dummy = torch.randn(1, 3, 32, 32)
    os.makedirs("export", exist_ok=True)

    torch.onnx.export(
        model, dummy, ONNX_FP32_PATH,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=17,
        dynamo=False 
    )
    size = os.path.getsize(ONNX_FP32_PATH)
    print(f"  Exported FP32 ONNX: {size / 1e6:.2f} MB")
    return model


# ─── Step 2: Dynamic INT8 Quantization ───────────────────────
def quantize_dynamic():
    print("\nStep 2: Dynamic INT8 quantization...")
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from onnxruntime.quantization.preprocess import quant_pre_process

    preprocessed = "export/resnet18_fp32_prep.onnx"
    quant_pre_process(ONNX_FP32_PATH, preprocessed)

    quantize_dynamic(
        preprocessed,
        ONNX_INT8_DYN_PATH,
        weight_type=QuantType.QUInt8
    )
    size = os.path.getsize(ONNX_INT8_DYN_PATH)
    print(f"  INT8 dynamic model: {size / 1e6:.2f} MB")

    if os.path.exists(preprocessed):
        os.remove(preprocessed)

# ─── Step 3: Static INT8 Quantization with calibration ───────
def quantize_static():
    print("\nStep 3: Static INT8 quantization (weights + activations)...")
    from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader
    from onnxruntime.quantization.preprocess import quant_pre_process

    class CifarCalibReader(CalibrationDataReader):
        def __init__(self, num_samples=100):
            # self.loader = get_test_loader(batch_size=1)
            self.loader = get_calibration_loader(batch_size=1)
            self.iter = iter(self.loader)
            self.count = 0
            self.num_samples = num_samples
            self.count = 0

        # def get_next(self):
        #     if self.count >= self.num_samples:
        #         return None
        #     try:
        #         inputs, _ = next(self.iter)
        #         self.count += 1
        #         return {'input': inputs.numpy()}
        #     except StopIteration:
        #         return None
        def get_next(self):
            if self.count >= self.num_samples:
                return None
            inputs, _ = next(self.iter)
            self.count += 1
            return {'input': inputs.numpy()}

    try:
        preprocessed = "export/resnet18_fp32_prep_static.onnx"
        quant_pre_process(ONNX_FP32_PATH, preprocessed)

        reader = CifarCalibReader(num_samples=100)
        quantize_static(
            preprocessed,
            ONNX_INT8_STATIC_PATH,
            calibration_data_reader=reader,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8
        )
        size = os.path.getsize(ONNX_INT8_STATIC_PATH)
        print(f"  INT8 static model: {size / 1e6:.2f} MB")

        if os.path.exists(preprocessed):
            os.remove(preprocessed)
        return True
    except Exception as e:
        print(f"  Static quantization failed: {e}")
        return False


# ─── Benchmarking ─────────────────────────────────────────────
def benchmark_onnx(model_path, label):
    import onnxruntime as ort

    print(f"\n{'─' * 50}")
    print(f"Benchmarking: {label}")
    print(f"{'─' * 50}")

    # Create session with CPU provider only
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1  # Single-threaded for fair comparison
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(model_path, opts, providers=['CPUExecutionProvider'])

    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 3, 32, 32).astype(np.float32)

    # Warmup
    print(f"  Warming up ({WARMUP_RUNS} runs)...")
    for _ in range(WARMUP_RUNS):
        session.run(None, {input_name: dummy})

    # Benchmark
    print(f"  Benchmarking ({BENCHMARK_RUNS} runs)...")
    times = []
    for _ in range(BENCHMARK_RUNS):
        start = time.perf_counter()
        session.run(None, {input_name: dummy})
        end = time.perf_counter()
        times.append((end - start) * 1000)

    times = np.array(times)
    print(f"  Mean:   {np.mean(times):.3f} ms")
    print(f"  Median: {np.median(times):.3f} ms")
    print(f"  Std:    {np.std(times):.3f} ms")
    print(f"  Min:    {np.min(times):.3f} ms")
    print(f"  P95:    {np.percentile(times, 95):.3f} ms")

    return np.median(times)


def evaluate_onnx_accuracy(model_path, label):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(model_path, opts, providers=['CPUExecutionProvider'])

    input_name = session.get_inputs()[0].name
    loader = get_test_loader(batch_size=1)
    

    correct = 0
    total = 0
    for inputs, targets in loader:
        outputs = session.run(None, {input_name: inputs.numpy()})
        pred = np.argmax(outputs[0], axis=1)
        correct += (pred == targets.numpy()).sum()
        total += targets.size(0)

    acc = 100.0 * correct / total
    print(f"  {label} accuracy: {acc:.2f}%")
    return acc


def get_calibration_loader(batch_size=1, num_samples=100):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(
        root='./data',
        train=True,   # 👈 THIS is the fix
        download=True,
        transform=transform
    )

    loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    return loader

def main():
    print("=" * 60)
    print("ONNX INT8 Quantization Benchmark")
    print("=" * 60)

    # Export and quantize
    export_to_onnx()
    quantize_dynamic()
    has_static = quantize_static()

    # ─── Benchmark all variants ───────────────────────────────
    print("\n" + "=" * 60)
    print("LATENCY BENCHMARKS (single-threaded CPU)")
    print("=" * 60)

    fp32_time = benchmark_onnx(ONNX_FP32_PATH, "ONNX FP32")
    int8_dyn_time = benchmark_onnx(ONNX_INT8_DYN_PATH, "ONNX INT8 Dynamic")

    int8_static_time = None
    if has_static and os.path.exists(ONNX_INT8_STATIC_PATH):
        int8_static_time = benchmark_onnx(ONNX_INT8_STATIC_PATH, "ONNX INT8 Static")

    # ─── Accuracy ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ACCURACY VERIFICATION")
    print("=" * 60)

    fp32_acc = evaluate_onnx_accuracy(ONNX_FP32_PATH, "ONNX FP32")
    int8_dyn_acc = evaluate_onnx_accuracy(ONNX_INT8_DYN_PATH, "ONNX INT8 Dynamic")

    int8_static_acc = None
    if has_static and os.path.exists(ONNX_INT8_STATIC_PATH):
        int8_static_acc = evaluate_onnx_accuracy(ONNX_INT8_STATIC_PATH, "ONNX INT8 Static")

    # ─── Model sizes ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL SIZES")
    print("=" * 60)
    print(f"  ONNX FP32:         {os.path.getsize(ONNX_FP32_PATH) / 1e6:.2f} MB")
    print(f"  ONNX INT8 Dynamic: {os.path.getsize(ONNX_INT8_DYN_PATH) / 1e6:.2f} MB")
    if has_static and os.path.exists(ONNX_INT8_STATIC_PATH):
        print(f"  ONNX INT8 Static:  {os.path.getsize(ONNX_INT8_STATIC_PATH) / 1e6:.2f} MB")
    print(f"  Our ternary (packed): ~3.50 MB")

    # ─── Summary comparison ───────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print(f"{'Method':<30} {'Latency (ms)':<15} {'Accuracy':<12} {'Size (MB)':<10}")
    print(f"{'-'*67}")
    print(f"{'ONNX FP32':<30} {fp32_time:<15.3f} {fp32_acc:<12.2f} {os.path.getsize(ONNX_FP32_PATH)/1e6:<10.2f}")
    print(f"{'ONNX INT8 Dynamic':<30} {int8_dyn_time:<15.3f} {int8_dyn_acc:<12.2f} {os.path.getsize(ONNX_INT8_DYN_PATH)/1e6:<10.2f}")
    if int8_static_time:
        print(f"{'ONNX INT8 Static':<30} {int8_static_time:<15.3f} {int8_static_acc:<12.2f} {os.path.getsize(ONNX_INT8_STATIC_PATH)/1e6:<10.2f}")
    print(f"{'Our Ternary (PyTorch CPU)':<30} {'46.630':<15} {'94.48':<12} {'3.50':<10}")
    print(f"{'Our Ternary (C++ SIMD)':<30} {'???':<15} {'94.48':<12} {'3.50':<10}")
    print(f"=" * 67)

    print("\nNote: ONNX benchmarks use single-threaded CPU for fair comparison.")
    print("Our ternary C++ engine is also single-threaded.")


if __name__ == "__main__":
    main()
