"""Calibrate static scales for INT8-resident inference.

Captures activation ranges at every boundary point:
- Post-ReLU (conv input) → unsigned uint8 [0, 255], scale = max/255
- Post-BN (conv output, before add/relu) → signed int8 [-128, 127], scale = absmax/127
- Post-add+ReLU (block output) → unsigned uint8 [0, 255], scale = max/255

Exports scales as JSON for the C++ engine.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from models.resnet_ternary import ternary_resnet18_cifar

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "resnet18_ternary_best.pth")

model = ternary_resnet18_cifar()
ckpt = torch.load(CKPT, map_location='cpu', weights_only=True)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
calibset = datasets.CIFAR10(root=DATA_DIR, train=True, download=False, transform=transform)
calibloader = torch.utils.data.DataLoader(calibset, batch_size=128, shuffle=False, num_workers=4)

# We need to trace through the model manually to capture ranges at every boundary
scales = {}

def update_range(name, tensor):
    """Track min/max for a tensor across calibration batches."""
    vmin = tensor.min().item()
    vmax = tensor.max().item()
    if name in scales:
        scales[name] = (min(scales[name][0], vmin), max(scales[name][1], vmax))
    else:
        scales[name] = (vmin, vmax)

print("Running calibration (100 batches)...")
with torch.no_grad():
    for batch_idx, (images, _) in enumerate(calibloader):
        if batch_idx >= 100:
            break

        # ─── Manual forward pass with range tracking ───
        x = images
        update_range("input", x)

        # Conv1 + BN1 + ReLU
        x = model.conv1(x)
        x = model.bn1(x)
        update_range("conv1_bn_out", x)  # post-BN, pre-ReLU (signed)
        x = F.relu(x)
        update_range("conv1_relu_out", x)  # post-ReLU (unsigned, conv input for next)

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

                # Conv2 + BN2 (no ReLU)
                out = block.conv2(out)
                out = block.bn2(out)
                update_range(f"{prefix}.conv2_bn_out", out)

                # Shortcut
                if len(block.shortcut) > 0:
                    identity = block.shortcut(identity)
                    update_range(f"{prefix}.shortcut_out", identity)

                # Add + ReLU
                out = out + identity
                update_range(f"{prefix}.add_out", out)
                out = F.relu(out)
                update_range(f"{prefix}.block_relu_out", out)

                x = out

        # Avgpool + FC
        x = model.avg_pool(x)
        update_range("avgpool_out", x)

    if (batch_idx + 1) % 20 == 0:
        print(f"  batch {batch_idx+1}")

# Compute scales
result = {}
for name, (vmin, vmax) in sorted(scales.items()):
    is_post_relu = "relu_out" in name
    if is_post_relu:
        # Unsigned: [0, max] → uint8 [0, 255]
        scale = vmax / 255.0 if vmax > 0 else 1.0
        result[name] = {"min": vmin, "max": vmax, "scale": scale, "unsigned": True}
    else:
        # Signed: [-absmax, absmax] → int8 [-128, 127]
        absmax = max(abs(vmin), abs(vmax))
        scale = absmax / 127.0 if absmax > 0 else 1.0
        result[name] = {"min": vmin, "max": vmax, "scale": scale, "unsigned": False}

print(f"\nCalibrated {len(result)} scale points:")
for name, info in sorted(result.items()):
    u = "uint8" if info["unsigned"] else "int8"
    print(f"  {name:40s} [{info['min']:8.4f}, {info['max']:8.4f}] scale={info['scale']:.6f} ({u})")

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static_scales.json")
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to {out_path}")
