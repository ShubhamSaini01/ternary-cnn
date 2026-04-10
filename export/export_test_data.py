"""Export CIFAR-10 test set as raw binary for C++ engine."""
import sys, os, struct
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from torchvision import datasets, transforms

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cifar10_test.bin")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = datasets.CIFAR10(root=DATA_DIR, train=False, download=False, transform=transform)

with open(OUT_PATH, 'wb') as f:
    n = len(testset)
    f.write(struct.pack('<I', n))
    for i in range(n):
        img, label = testset[i]
        f.write(struct.pack('<I', label))
        f.write(img.numpy().astype(np.float32).tobytes())

print(f"Exported {n} test images to {OUT_PATH}")
print(f"Size: {os.path.getsize(OUT_PATH) / 1e6:.2f} MB")
