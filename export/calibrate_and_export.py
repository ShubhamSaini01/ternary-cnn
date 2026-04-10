"""Calibrate static INT8 activation scales and export ternary weights for C++ engine.

Outputs:
  export/ternary_resnet18.bin  — packed binary weights + BN params + scales
  export/activation_scales.json — per-layer activation scales from calibration
"""

import sys
import os
import json
import struct

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from models.resnet_ternary import ternary_resnet18_cifar
from models.ternary_conv import TernaryConv2d, TernaryQuantize

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CKPT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "resnet18_ternary_best.pth")
EXPORT_DIR = os.path.dirname(os.path.abspath(__file__))

CALIB_BATCHES = 100


def calibrate(model):
    """Run calibration to get per-layer activation min/max for static quantization.

    Hooks after every conv+bn+relu boundary to capture output ranges.
    Returns ordered list of (layer_name, scale, zero_point) tuples.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    calibset = datasets.CIFAR10(root=DATA_DIR, train=True, download=False, transform=transform)
    calibloader = torch.utils.data.DataLoader(calibset, batch_size=128, shuffle=False, num_workers=4)

    model.eval()
    layer_stats = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, inp, out):
            with torch.no_grad():
                omin = out.min().item()
                omax = out.max().item()
                if name in layer_stats:
                    layer_stats[name] = (
                        min(layer_stats[name][0], omin),
                        max(layer_stats[name][1], omax),
                    )
                else:
                    layer_stats[name] = (omin, omax)
        return hook_fn

    # Hook after every BatchNorm2d (which follows conv in our architecture)
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Also capture model input range
    def input_hook(module, inp, out):
        x = inp[0]
        with torch.no_grad():
            layer_stats["_input"] = (
                min(layer_stats.get("_input", (0, 0))[0], x.min().item()),
                max(layer_stats.get("_input", (0, 0))[1], x.max().item()),
            )
    hooks.append(model.conv1.register_forward_hook(input_hook))

    with torch.no_grad():
        for i, (inputs, _) in enumerate(calibloader):
            if i >= CALIB_BATCHES:
                break
            model(inputs)

    for h in hooks:
        h.remove()

    # Compute scales
    # After ReLU: range is [0, max] -> unsigned uint8 [0, 255], scale = max/255
    # Input/signed: range is [min, max] -> signed int8 [-128, 127]
    scales = {}
    for name, (vmin, vmax) in sorted(layer_stats.items()):
        if name == "_input":
            # Signed: symmetric around 0
            absmax = max(abs(vmin), abs(vmax))
            scales[name] = {"min": vmin, "max": vmax, "scale": absmax / 127.0 if absmax > 0 else 1.0, "signed": True}
        else:
            # Post-BN can be negative (before ReLU) or positive (after ReLU in residual)
            # Use full range symmetric quantization
            absmax = max(abs(vmin), abs(vmax))
            scales[name] = {"min": vmin, "max": vmax, "scale": absmax / 127.0 if absmax > 0 else 1.0, "signed": True}

    print(f"Calibrated {len(scales)} layers:")
    for name, info in sorted(scales.items()):
        print(f"  {name}: [{info['min']:.4f}, {info['max']:.4f}] scale={info['scale']:.6f}")

    return scales


def pack_ternary_i2s(ternary_weights):
    """Pack ternary weights {-1, 0, +1} into I2_S 2-bit format.

    Encoding per weight (2 bits):
      +1 -> 0b01
       0 -> 0b00
      -1 -> 0b11

    Packs 4 weights per byte, MSB first.
    Returns numpy uint8 array.
    """
    flat = ternary_weights.flatten().numpy().astype(np.int8)
    # Encode: +1->01, 0->00, -1->11
    encoded = np.zeros_like(flat, dtype=np.uint8)
    encoded[flat == 1] = 0b01
    encoded[flat == -1] = 0b11
    # encoded[flat == 0] = 0b00  (already zero)

    # Pad to multiple of 4
    pad_len = (4 - len(encoded) % 4) % 4
    if pad_len:
        encoded = np.concatenate([encoded, np.zeros(pad_len, dtype=np.uint8)])

    # Pack 4 values per byte
    packed = np.zeros(len(encoded) // 4, dtype=np.uint8)
    for i in range(4):
        packed |= encoded[i::4] << (6 - 2 * i)

    return packed


def export_model(model, scales):
    """Export model to binary format for C++ engine.

    Format:
      Header:
        magic: uint32 = 0x54524E59 ("TRNY")
        version: uint32 = 1
        num_layers: uint32

      Per layer (ordered by execution):
        layer_type: uint8 (0=conv_fp32, 1=conv_ternary, 2=bn, 3=fc, 4=avgpool)
        name_len: uint16
        name: char[name_len]
        ... type-specific data ...

      Conv ternary:
        out_channels, in_channels, kH, kW: uint16 each
        stride, padding: uint16 each
        alpha: float32[out_channels]
        packed_weights: uint8[ceil(OC*IC*kH*kW*2/8)]

      Conv FP32:
        out_channels, in_channels, kH, kW: uint16 each
        stride, padding: uint16 each
        weights: float32[OC*IC*kH*kW]

      BatchNorm:
        num_features: uint16
        weight, bias, running_mean, running_var: float32[num_features] each
        eps: float32

      FC:
        in_features, out_features: uint16 each
        weight: float32[out*in]
        bias: float32[out]

      Activation scale (appended after each conv+bn group):
        scale: float32
    """
    out_path = os.path.join(EXPORT_DIR, "ternary_resnet18.bin")
    model.eval()

    with open(out_path, "wb") as f:
        # Magic + version
        f.write(struct.pack("<I", 0x54524E59))
        f.write(struct.pack("<I", 1))

        # We'll write layers in execution order
        # Collect all layers we need to export
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
                # Shortcut
                if len(block.shortcut) > 0:
                    layers.append(("conv_fp32", f"{prefix}.shortcut.0", block.shortcut[0]))
                    layers.append(("bn", f"{prefix}.shortcut.1", block.shortcut[1]))

        # Avgpool + FC
        layers.append(("avgpool", "avg_pool", model.avg_pool))
        layers.append(("fc", "fc", model.fc))

        # Write num_layers
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
                # Get ternary weights and alpha
                ternary, alpha = TernaryQuantize.apply(module.weight.data)
                oc, ic, kh, kw = module.weight.shape
                stride = module.stride if isinstance(module.stride, int) else module.stride[0]
                padding = module.padding if isinstance(module.padding, int) else module.padding[0]
                f.write(struct.pack("<HHHH", oc, ic, kh, kw))
                f.write(struct.pack("<HH", stride, padding))
                # Alpha per output channel
                f.write(alpha.cpu().numpy().astype(np.float32).tobytes())
                # Packed ternary weights
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
                # Write activation scale for this BN's output
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
    print("Loading ternary model...")
    model = ternary_resnet18_cifar()
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint (acc: {ckpt['test_acc']:.2f}%)")

    print("\n── Calibrating activation scales ──")
    scales = calibrate(model)

    # Save scales as JSON for reference
    scales_path = os.path.join(EXPORT_DIR, "activation_scales.json")
    with open(scales_path, "w") as f:
        json.dump(scales, f, indent=2)
    print(f"Scales saved to {scales_path}")

    print("\n── Exporting model ──")
    export_model(model, scales)
