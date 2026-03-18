"""
Export ternary ResNet-18 weights for C++ inference engine.
Usage: python export/export_weights.py

Key optimization: α is folded into BN scale, so C++ only needs:
  output = fused_scale * ternary_conv_output + fused_bias

Binary format (sequential, no padding):
  [HEADER]
    magic: uint32 = 0x54524E59 ("TRNY")
    num_layers: uint32
  [Per layer]
    layer_type: uint8 (0=fp32_conv, 1=ternary_conv, 2=fc)
    out_channels: uint32
    in_channels: uint32
    kernel_h: uint32
    kernel_w: uint32
    stride: uint32
    padding: uint32
    has_bn: uint8 (1 if fused scale/bias follow)

    if fp32_conv:
      weights: float32[out_c * in_c * kH * kW]
      if has_bn: fused_scale, fused_bias: float32[out_c] each

    if ternary_conv:
      mask_pos: uint8[ceil(out_c * in_c * kH * kW / 8)]
      mask_neg: uint8[ceil(out_c * in_c * kH * kW / 8)]
      fused_scale: float32[out_c]  (= bn_scale * alpha)
      fused_bias: float32[out_c]   (= bn_bias)

    if fc:
      weights: float32[out_features * in_features]
      bias: float32[out_features]
"""

import sys
import os
import struct
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.resnet_ternary import ternary_resnet18_cifar
from models.ternary_conv import TernaryConv2d

CHECKPOINT = "checkpoints/resnet18_ternary_best.pth"
OUTPUT = "export/ternary_resnet18.bin"
MAGIC = 0x54524E59  # "TRNY"


def fuse_bn(bn):
    """
    Fuse BatchNorm into scale and bias.
    BN(x) = (gamma / sqrt(var + eps)) * x + (beta - gamma * mean / sqrt(var + eps))
    Returns (scale, bias) as float32 arrays.
    """
    gamma = bn.weight.data.cpu().numpy()
    beta = bn.bias.data.cpu().numpy()
    mean = bn.running_mean.data.cpu().numpy()
    var = bn.running_var.data.cpu().numpy()
    eps = bn.eps

    scale = gamma / np.sqrt(var + eps)
    bias = beta - gamma * mean / np.sqrt(var + eps)
    return scale.astype(np.float32), bias.astype(np.float32)


def bitpack_ternary(weights):
    """
    Bitpack ternary weights into two uint8 arrays.
    mask_pos: bit=1 where weight=+1
    mask_neg: bit=1 where weight=-1
    """
    flat = weights.flatten()
    n = len(flat)
    padded_n = ((n + 7) // 8) * 8
    padded = np.zeros(padded_n, dtype=np.float32)
    padded[:n] = flat

    num_bytes = padded_n // 8
    mask_pos = np.zeros(num_bytes, dtype=np.uint8)
    mask_neg = np.zeros(num_bytes, dtype=np.uint8)

    for i in range(num_bytes):
        for bit in range(8):
            idx = i * 8 + bit
            if padded[idx] > 0.5:
                mask_pos[i] |= (1 << bit)
            elif padded[idx] < -0.5:
                mask_neg[i] |= (1 << bit)

    return mask_pos, mask_neg


def compute_ternary_and_alpha(fp_weights):
    """
    Quantize FP32 shadow weights to ternary and compute alpha.
    Mirrors TernaryQuantize.forward() logic.
    """
    w = fp_weights.cpu().numpy()
    abs_w = np.abs(w)

    # Threshold per output channel
    delta = 0.7 * abs_w.mean(axis=(1, 2, 3), keepdims=True)

    # Ternary assignment
    ternary = np.zeros_like(w)
    ternary[w > delta] = 1.0
    ternary[w < -delta] = -1.0

    # Alpha per channel
    alpha = np.zeros(w.shape[0], dtype=np.float32)
    for i in range(w.shape[0]):
        mask = ternary[i] != 0
        if mask.any():
            alpha[i] = abs_w[i][mask].mean()

    return ternary, alpha


def write_layer(f, layer_type, out_c, in_c, kH, kW, stride, padding, has_bn,
                weights_data=None, fused_scale=None, fused_bias=None,
                mask_pos=None, mask_neg=None, fc_bias=None):
    """Write a single layer to the binary file."""
    f.write(struct.pack('B', layer_type))
    f.write(struct.pack('I', out_c))
    f.write(struct.pack('I', in_c))
    f.write(struct.pack('I', kH))
    f.write(struct.pack('I', kW))
    f.write(struct.pack('I', stride))
    f.write(struct.pack('I', padding))
    f.write(struct.pack('B', 1 if has_bn else 0))

    if layer_type == 0:  # fp32_conv
        f.write(weights_data.tobytes())
        if has_bn:
            f.write(fused_scale.tobytes())
            f.write(fused_bias.tobytes())

    elif layer_type == 1:  # ternary_conv
        f.write(mask_pos.tobytes())
        f.write(mask_neg.tobytes())
        # fused_scale already has alpha folded in
        f.write(fused_scale.tobytes())
        f.write(fused_bias.tobytes())

    elif layer_type == 2:  # fc
        f.write(weights_data.tobytes())
        f.write(fc_bias.tobytes())


def export_model(model):
    """Export the full model to binary format."""
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    layers = []

    # === Layer 0: First conv (FP32) + BN ===
    w = model.conv1.weight.data.cpu().numpy().astype(np.float32)
    bn_scale, bn_bias = fuse_bn(model.bn1)
    layers.append(('fp32_conv', model.conv1, w, bn_scale, bn_bias))
    print(f"  conv1: FP32 {w.shape}, BN fused")

    # === ResNet blocks ===
    for layer_idx, layer in enumerate([model.layer1, model.layer2, model.layer3, model.layer4]):
        for block_idx, block in enumerate(layer):
            prefix = f"  layer{layer_idx+1}.block{block_idx}"

            # Conv1 (ternary) — fold alpha into BN scale
            ternary_w, alpha = compute_ternary_and_alpha(block.conv1.weight.data)
            mask_pos, mask_neg = bitpack_ternary(ternary_w)
            bn_scale, bn_bias = fuse_bn(block.bn1)
            fused_scale = (bn_scale * alpha).astype(np.float32)
            layers.append(('ternary_conv', block.conv1,
                           ternary_w, fused_scale, bn_bias, mask_pos, mask_neg))
            print(f"{prefix}.conv1: Ternary {ternary_w.shape}, "
                  f"packed {len(mask_pos)} bytes, "
                  f"alpha [{alpha.min():.6f}, {alpha.max():.6f}], "
                  f"fused_scale [{fused_scale.min():.4f}, {fused_scale.max():.4f}]")

            # Conv2 (ternary) — fold alpha into BN scale
            ternary_w, alpha = compute_ternary_and_alpha(block.conv2.weight.data)
            mask_pos, mask_neg = bitpack_ternary(ternary_w)
            bn_scale, bn_bias = fuse_bn(block.bn2)
            fused_scale = (bn_scale * alpha).astype(np.float32)
            layers.append(('ternary_conv', block.conv2,
                           ternary_w, fused_scale, bn_bias, mask_pos, mask_neg))
            print(f"{prefix}.conv2: Ternary {ternary_w.shape}, "
                  f"packed {len(mask_pos)} bytes, "
                  f"fused_scale [{fused_scale.min():.4f}, {fused_scale.max():.4f}]")

            # Shortcut (if exists, FP32)
            if len(block.shortcut) > 0:
                sc_conv = block.shortcut[0]
                sc_bn = block.shortcut[1]
                w = sc_conv.weight.data.cpu().numpy().astype(np.float32)
                bn_scale, bn_bias = fuse_bn(sc_bn)
                layers.append(('fp32_conv', sc_conv, w, bn_scale, bn_bias))
                print(f"{prefix}.shortcut: FP32 {w.shape}, BN fused")

    # === FC layer ===
    fc_w = model.fc.weight.data.cpu().numpy().astype(np.float32)
    fc_b = model.fc.bias.data.cpu().numpy().astype(np.float32)
    layers.append(('fc', model.fc, fc_w, fc_b))
    print(f"  fc: FP32 {fc_w.shape}")

    # === Write binary file ===
    with open(OUTPUT, 'wb') as f:
        # Header
        f.write(struct.pack('I', MAGIC))
        f.write(struct.pack('I', len(layers)))

        for entry in layers:
            if entry[0] == 'fp32_conv':
                _, conv, w, fs, fb = entry
                out_c, in_c, kH, kW = w.shape
                stride = conv.stride[0] if isinstance(conv.stride, tuple) else conv.stride
                padding = conv.padding[0] if isinstance(conv.padding, tuple) else conv.padding
                write_layer(f, 0, out_c, in_c, kH, kW, stride, padding, True,
                            weights_data=w, fused_scale=fs, fused_bias=fb)

            elif entry[0] == 'ternary_conv':
                _, conv, tw, fs, fb, mp, mn = entry
                out_c, in_c, kH, kW = tw.shape
                stride = conv.stride if isinstance(conv.stride, int) else conv.stride[0]
                padding = conv.padding if isinstance(conv.padding, int) else conv.padding[0]
                write_layer(f, 1, out_c, in_c, kH, kW, stride, padding, True,
                            fused_scale=fs, fused_bias=fb,
                            mask_pos=mp, mask_neg=mn)

            elif entry[0] == 'fc':
                _, fc, w, b = entry
                out_f, in_f = w.shape
                write_layer(f, 2, out_f, in_f, 1, 1, 1, 0, False,
                            weights_data=w, fc_bias=b)

        total_bytes = f.tell()

    print(f"\nExported to: {OUTPUT}")
    print(f"Total file size: {total_bytes / 1024:.1f} KB ({total_bytes / 1e6:.2f} MB)")


def main():
    print("=" * 60)
    print("Exporting Ternary ResNet-18 for C++ Inference")
    print("=" * 60)

    print(f"\nLoading checkpoint: {CHECKPOINT}")
    model = ternary_resnet18_cifar(num_classes=10)
    ckpt = torch.load(CHECKPOINT, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Accuracy: {ckpt.get('test_acc', 'N/A')}%\n")

    print("Exporting layers:")
    export_model(model)
    print("\nDone! Binary file ready for C++ engine.")


if __name__ == "__main__":
    main()
