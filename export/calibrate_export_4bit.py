"""Calibrate 4-bit activation scales and export ternary weights for the 4-bit QAT model.

Uses max/15 for post-ReLU scales (4-bit range [0,15]) instead of max/255.
Exports weights in same binary format as calibrate_and_export.py.

Outputs:
  export/ternary_resnet18_4bit.bin     — packed binary weights + BN params
  export/static_scales_4bit.json       — per-layer 4-bit activation scales
"""

import sys
import os
import json
import struct

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms

# Import the 4-bit QAT model and its TernaryQuantize
from train_ternary_4bit import TernaryResNet18_4bit, TernaryQuantize

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
EXPORT_DIR = os.path.dirname(os.path.abspath(__file__))

# Try both checkpoint locations
CKPT_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "resnet18_ternary_4bit_best.pth"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train", "checkpoints", "resnet18_ternary_4bit_best.pth"),
]

CALIB_BATCHES = 100


def find_checkpoint():
    for path in CKPT_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No 4-bit QAT checkpoint found. Tried: {CKPT_CANDIDATES}")


def calibrate(model):
    """Run calibration to capture activation ranges at every boundary.
    Uses max/15 for unsigned post-ReLU (4-bit) scales.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    calibset = datasets.CIFAR10(root=DATA_DIR, train=True, download=False, transform=transform)
    calibloader = torch.utils.data.DataLoader(calibset, batch_size=128, shuffle=False, num_workers=4)

    model.eval()
    scales = {}

    def update_range(name, tensor):
        vmin = tensor.min().item()
        vmax = tensor.max().item()
        if name in scales:
            scales[name] = (min(scales[name][0], vmin), max(scales[name][1], vmax))
        else:
            scales[name] = (vmin, vmax)

    print("Running calibration (100 batches)...")
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(calibloader):
            if batch_idx >= CALIB_BATCHES:
                break

            x = images

            # Conv1 + BN1 + ReLU (first conv is FP32)
            x = model.conv1(x)
            x = model.bn1(x)
            x = F.relu(x)
            update_range("conv1_relu_out", x)

            # ResNet blocks
            for layer_idx, layer in enumerate([model.layer1, model.layer2, model.layer3, model.layer4]):
                for blk_idx, block in enumerate(layer):
                    identity = x
                    prefix = f"layer{layer_idx+1}.{blk_idx}"

                    # Conv1 + BN1 + ReLU
                    out = block.conv1(x)
                    out = block.bn1(out)
                    update_range(f"{prefix}.conv1_bn_out", out)
                    out = F.relu(out)
                    update_range(f"{prefix}.conv1_relu_out", out)

                    # Conv2 + BN2
                    out = block.conv2(out)
                    out = block.bn2(out)
                    update_range(f"{prefix}.conv2_bn_out", out)

                    # Shortcut
                    if len(block.shortcut) > 0:
                        identity = block.shortcut(identity)
                        update_range(f"{prefix}.shortcut_out", identity)

                    # Add + ReLU
                    out = out + identity
                    out = F.relu(out)
                    update_range(f"{prefix}.block_relu_out", out)

                    x = out

            if (batch_idx + 1) % 20 == 0:
                print(f"  batch {batch_idx+1}")

    # Compute scales: max/15 for unsigned, absmax/127 for signed
    result = {}
    for name, (vmin, vmax) in sorted(scales.items()):
        is_post_relu = "relu_out" in name
        if is_post_relu:
            # 4-bit unsigned: [0, max] → [0, 15]
            scale = vmax / 15.0 if vmax > 0 else 1.0
            result[name] = {"min": vmin, "max": vmax, "scale": scale, "unsigned": True}
        else:
            # Signed (for reference, not directly used in engine)
            absmax = max(abs(vmin), abs(vmax))
            scale = absmax / 127.0 if absmax > 0 else 1.0
            result[name] = {"min": vmin, "max": vmax, "scale": scale, "unsigned": False}

    print(f"\nCalibrated {len(result)} scale points:")
    for name, info in sorted(result.items()):
        u = "uint4" if info["unsigned"] else "int8"
        print(f"  {name:40s} [{info['min']:8.4f}, {info['max']:8.4f}] scale={info['scale']:.6f} ({u})")

    return result


def pack_ternary_i2s(ternary_weights):
    """Pack ternary weights {-1, 0, +1} into I2_S 2-bit format."""
    flat = ternary_weights.flatten().numpy().astype(np.int8)
    encoded = np.zeros_like(flat, dtype=np.uint8)
    encoded[flat == 1] = 0b01
    encoded[flat == -1] = 0b11

    pad_len = (4 - len(encoded) % 4) % 4
    if pad_len:
        encoded = np.concatenate([encoded, np.zeros(pad_len, dtype=np.uint8)])

    packed = np.zeros(len(encoded) // 4, dtype=np.uint8)
    for i in range(4):
        packed |= encoded[i::4] << (6 - 2 * i)

    return packed


def export_model(model, scales):
    """Export model to binary format (same format as original engine)."""
    out_path = os.path.join(EXPORT_DIR, "ternary_resnet18_4bit.bin")
    model.eval()

    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", 0x54524E59))  # magic
        f.write(struct.pack("<I", 1))            # version

        layers = []

        # First conv (FP32) + BN
        layers.append(("conv_fp32", "conv1", model.conv1))
        layers.append(("bn", "bn1", model.bn1))

        # ResNet blocks
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            layer = getattr(model, layer_name)
            for block_idx, block in enumerate(layer):
                prefix = f"{layer_name}.{block_idx}"
                layers.append(("conv_ternary", f"{prefix}.conv1", block.conv1))
                layers.append(("bn", f"{prefix}.bn1", block.bn1))
                layers.append(("conv_ternary", f"{prefix}.conv2", block.conv2))
                layers.append(("bn", f"{prefix}.bn2", block.bn2))
                if len(block.shortcut) > 0:
                    layers.append(("conv_fp32", f"{prefix}.shortcut.0", block.shortcut[0]))
                    layers.append(("bn", f"{prefix}.shortcut.1", block.shortcut[1]))

        layers.append(("avgpool", "avg_pool", model.avg_pool))
        layers.append(("fc", "fc", model.fc))

        f.write(struct.pack("<I", len(layers)))

        for ltype, name, module in layers:
            name_bytes = name.encode("utf-8")

            if ltype == "conv_fp32":
                f.write(struct.pack("<B", 0))
                f.write(struct.pack("<H", len(name_bytes)))
                f.write(name_bytes)
                w = module.weight.data
                oc, ic, kh, kw = w.shape
                stride = module.stride if isinstance(module.stride, int) else module.stride[0]
                padding = module.padding if isinstance(module.padding, int) else module.padding[0]
                f.write(struct.pack("<HHHH", oc, ic, kh, kw))
                f.write(struct.pack("<HH", stride, padding))
                f.write(w.cpu().numpy().astype(np.float32).tobytes())

            elif ltype == "conv_ternary":
                f.write(struct.pack("<B", 1))
                f.write(struct.pack("<H", len(name_bytes)))
                f.write(name_bytes)
                ternary, alpha = TernaryQuantize.apply(module.weight.data)
                oc, ic, kh, kw = module.weight.shape
                stride = module.stride if isinstance(module.stride, int) else module.stride[0]
                padding = module.padding if isinstance(module.padding, int) else module.padding[0]
                f.write(struct.pack("<HHHH", oc, ic, kh, kw))
                f.write(struct.pack("<HH", stride, padding))
                f.write(alpha.cpu().numpy().astype(np.float32).tobytes())
                packed = pack_ternary_i2s(ternary)
                f.write(struct.pack("<I", len(packed)))
                f.write(packed.tobytes())

            elif ltype == "bn":
                f.write(struct.pack("<B", 2))
                f.write(struct.pack("<H", len(name_bytes)))
                f.write(name_bytes)
                nf = module.num_features
                f.write(struct.pack("<H", nf))
                f.write(module.weight.data.cpu().numpy().astype(np.float32).tobytes())
                f.write(module.bias.data.cpu().numpy().astype(np.float32).tobytes())
                f.write(module.running_mean.data.cpu().numpy().astype(np.float32).tobytes())
                f.write(module.running_var.data.cpu().numpy().astype(np.float32).tobytes())
                f.write(struct.pack("<f", module.eps))
                scale_info = scales.get(name, None)
                if scale_info:
                    f.write(struct.pack("<f", scale_info["scale"]))
                else:
                    f.write(struct.pack("<f", 1.0))

            elif ltype == "avgpool":
                f.write(struct.pack("<B", 4))
                f.write(struct.pack("<H", len(name_bytes)))
                f.write(name_bytes)

            elif ltype == "fc":
                f.write(struct.pack("<B", 3))
                f.write(struct.pack("<H", len(name_bytes)))
                f.write(name_bytes)
                inf = module.in_features
                outf = module.out_features
                f.write(struct.pack("<HH", inf, outf))
                f.write(module.weight.data.cpu().numpy().astype(np.float32).tobytes())
                f.write(module.bias.data.cpu().numpy().astype(np.float32).tobytes())

    file_size = os.path.getsize(out_path)
    print(f"\nExported to {out_path} ({file_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    ckpt_path = find_checkpoint()
    print(f"Loading 4-bit QAT model from {ckpt_path}...")
    model = TernaryResNet18_4bit(num_classes=10)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint (acc: {ckpt.get('test_acc', 'N/A')}%)")

    print("\n── Calibrating 4-bit activation scales ──")
    scales = calibrate(model)

    # Save scales JSON
    scales_path = os.path.join(EXPORT_DIR, "static_scales_4bit.json")
    with open(scales_path, "w") as f:
        json.dump(scales, f, indent=2)
    print(f"Scales saved to {scales_path}")

    print("\n── Exporting model ──")
    export_model(model, scales)
