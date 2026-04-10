"""Debug: run PyTorch layer by layer and dump intermediate outputs."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from models.resnet_ternary import ternary_resnet18_cifar
from models.ternary_conv import TernaryQuantize

CKPT = "checkpoints/resnet18_ternary_best.pth"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

model = ternary_resnet18_cifar()
ckpt = torch.load(CKPT, map_location='cpu', weights_only=True)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

img, label = testset[0]
x = img.unsqueeze(0)

print(f"Input: shape={x.shape}, min={x.min():.4f}, max={x.max():.4f}")

# Conv1 + BN1 + ReLU
with torch.no_grad():
    # Conv1 is FP32
    out = model.conv1(x)
    print(f"After conv1: min={out.min():.4f}, max={out.max():.4f}, mean={out.mean():.4f}")
    out = model.bn1(out)
    print(f"After bn1: min={out.min():.4f}, max={out.max():.4f}, mean={out.mean():.4f}")
    out = F.relu(out)
    print(f"After relu: min={out.min():.4f}, max={out.max():.4f}, mean={out.mean():.4f}")

    # Check ternary weights for first block
    blk = model.layer1[0]
    ternary_w, alpha = TernaryQuantize.apply(blk.conv1.weight)
    print(f"\nlayer1.0.conv1 ternary weights: unique={torch.unique(ternary_w).tolist()}, alpha[0]={alpha[0].item():.6f}")
    print(f"  weight stats: nnz={ternary_w.count_nonzero().item()}/{ternary_w.numel()}, "
          f"+1={( ternary_w==1).sum().item()}, -1={(ternary_w==-1).sum().item()}, 0={(ternary_w==0).sum().item()}")

    # Full forward
    final = model(x)
    pred = final.argmax(1).item()
    print(f"\nFinal output: {final[0][:5].tolist()}")
    print(f"Prediction: {pred}, Label: {label}")

    # Check fused BN for conv1
    bn = model.bn1
    std_inv = 1.0 / torch.sqrt(bn.running_var + bn.eps)
    fused_s = bn.weight * std_inv
    fused_b = bn.bias - bn.running_mean * fused_s
    print(f"\nFused BN1: scale[0]={fused_s[0]:.6f}, bias[0]={fused_b[0]:.6f}")

    # Verify fused conv1
    w = model.conv1.weight.data.clone()
    for oc in range(w.shape[0]):
        w[oc] *= fused_s[oc]
    fused_out = F.conv2d(x, w, None, stride=1, padding=1)
    for oc in range(fused_out.shape[1]):
        fused_out[0, oc] += fused_b[oc]
    print(f"Fused conv1 output: min={fused_out.min():.4f}, max={fused_out.max():.4f}")
    print(f"Match unfused? max_diff={( fused_out - (model.bn1(model.conv1(x)))).abs().max():.8f}")
