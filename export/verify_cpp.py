"""Generate a few test vectors and reference outputs to verify C++ engine correctness."""
import sys, os, struct
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torchvision import datasets, transforms
from models.resnet_ternary import ternary_resnet18_cifar

CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "resnet18_ternary_best.pth")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

model = ternary_resnet18_cifar()
ckpt = torch.load(CKPT, map_location='cpu', weights_only=True)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Save 10 test images as raw binary + print PyTorch predictions
with open('export/test_vectors.bin', 'wb') as f:
    f.write(struct.pack('<I', 10))
    for i in range(10):
        img, label = testset[i]
        f.write(struct.pack('<I', label))
        f.write(img.numpy().astype(np.float32).tobytes())

        with torch.no_grad():
            out = model(img.unsqueeze(0))
            pred = out.argmax(1).item()
            probs = out[0].numpy()

        print(f"Image {i}: label={label} pred={pred} logits=[{', '.join(f'{v:.4f}' for v in probs[:5])}...]")
