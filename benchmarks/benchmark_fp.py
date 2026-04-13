"""
Benchmark CPU inference for the FP32 baseline ResNet-18.
Usage: python benchmarks/benchmark_fp.py

Measures:
- Single image latency
- Batch throughput
- Model accuracy verification
- Memory footprint
"""

import sys
import os
import time
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.profiler import profile, ProfilerActivity
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.resnet_fp import resnet18_cifar


# ─── Config ───────────────────────────────────────────────────
CHECKPOINT = "checkpoints/resnet18_fp_best.pth"
BATCH_SIZE = 1       # Single image for latency measurement
WARMUP_RUNS = 50     # Warm up CPU caches
BENCHMARK_RUNS = 200 # Number of inference runs to average
DEVICE = "cpu"       # Always CPU — this is what we compare against C++
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

def benchmark_single_inference(model, input_tensor):
    model.eval()

    # Warmup (keep this)
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model(input_tensor)

    # 🔥 PROFILING (THIS IS THE NEW PART)
    print("\n[Profiling single inference]\n")

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True
    ) as prof:
        with torch.no_grad():
            _ = model(input_tensor)

    print(prof.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=20
    ))

    # Keep your original timing (optional but useful)
    times = []
    with torch.no_grad():
        for _ in range(BENCHMARK_RUNS):
            start = time.perf_counter()
            _ = model(input_tensor)
            end = time.perf_counter()
            times.append((end - start) * 1000)

    times = np.array(times)

    return {
        'mean_ms': np.mean(times),
        'median_ms': np.median(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'p95_ms': np.percentile(times, 95),
        'p99_ms': np.percentile(times, 99),
    }
# def benchmark_single_inference(model, input_tensor):
#     """Measure single-image inference latency."""
#     model.eval()

#     # Warmup — fill CPU caches, JIT compile if applicable
#     print(f"Warming up ({WARMUP_RUNS} runs)...")
#     with torch.no_grad():
#         for _ in range(WARMUP_RUNS):
#             _ = model(input_tensor)

#     # Benchmark
#     print(f"Benchmarking ({BENCHMARK_RUNS} runs)...")
#     times = []
#     with torch.no_grad():
#         for _ in range(BENCHMARK_RUNS):
#             start = time.perf_counter()
#             _ = model(input_tensor)
#             end = time.perf_counter()
#             times.append((end - start) * 1000)  # ms

#     times = np.array(times)
#     return {
#         'mean_ms': np.mean(times),
#         'median_ms': np.median(times),
#         'std_ms': np.std(times),
#         'min_ms': np.min(times),
#         'max_ms': np.max(times),
#         'p95_ms': np.percentile(times, 95),
#         'p99_ms': np.percentile(times, 99),
#     }


def benchmark_batch_throughput(model, loader, num_batches=50):
    """Measure batch throughput."""
    model.eval()
    total_images = 0
    total_time = 0

    with torch.no_grad():
        for i, (inputs, _) in enumerate(loader):
            if i >= num_batches:
                break
            start = time.perf_counter()
            _ = model(inputs)
            end = time.perf_counter()
            total_time += (end - start)
            total_images += inputs.size(0)

    return {
        'total_images': total_images,
        'total_time_s': total_time,
        'images_per_sec': total_images / total_time,
        'ms_per_image': (total_time / total_images) * 1000,
    }


@torch.no_grad()
def verify_accuracy(model, loader):
    """Verify model accuracy matches training results."""
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in loader:
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


def get_model_size(model):
    """Calculate model memory footprint."""
    total_params = sum(p.numel() for p in model.parameters())
    total_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_params, total_size_bytes


def main():
    print("=" * 60)
    print("FP32 Baseline — CPU Inference Benchmark")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from {CHECKPOINT}...")
    model = resnet18_cifar(num_classes=10)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    training_acc = ckpt.get('test_acc', 'N/A')
    print(f"Training best accuracy: {training_acc}")

    # Model size
    total_params, total_bytes = get_model_size(model)
    print(f"Parameters: {total_params:,}")
    print(f"Model size: {total_bytes / 1e6:.2f} MB")

    # Create test input (single CIFAR-10 image: 3x32x32)
    dummy_input = torch.randn(1, 3, 32, 32)

    # Single inference latency
    print(f"\n{'─' * 40}")
    print("Single Image Latency (batch_size=1)")
    print(f"{'─' * 40}")
    latency = benchmark_single_inference(model, dummy_input)
    print(f"  Mean:   {latency['mean_ms']:.3f} ms")
    print(f"  Median: {latency['median_ms']:.3f} ms")
    print(f"  Std:    {latency['std_ms']:.3f} ms")
    print(f"  Min:    {latency['min_ms']:.3f} ms")
    print(f"  Max:    {latency['max_ms']:.3f} ms")
    print(f"  P95:    {latency['p95_ms']:.3f} ms")
    print(f"  P99:    {latency['p99_ms']:.3f} ms")

    # Batch throughput
    print(f"\n{'─' * 40}")
    print("Batch Throughput (batch_size=128)")
    print(f"{'─' * 40}")
    batch_loader = get_test_loader(batch_size=128)
    throughput = benchmark_batch_throughput(model, batch_loader)
    print(f"  Images/sec: {throughput['images_per_sec']:.1f}")
    print(f"  ms/image:   {throughput['ms_per_image']:.3f} ms")

    # Verify accuracy
    print(f"\n{'─' * 40}")
    print("Accuracy Verification")
    print(f"{'─' * 40}")
    test_loader = get_test_loader(batch_size=128)
    acc = verify_accuracy(model, test_loader)
    print(f"  Test accuracy: {acc:.2f}%")
    print(f"  Training reported: {training_acc}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY — FP32 Baseline")
    print(f"{'=' * 60}")
    print(f"  Accuracy:         {acc:.2f}%")
    print(f"  Model size:       {total_bytes / 1e6:.2f} MB")
    print(f"  Parameters:       {total_params:,}")
    print(f"  Latency (median): {latency['median_ms']:.3f} ms")
    print(f"  Throughput:       {throughput['images_per_sec']:.1f} img/s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
