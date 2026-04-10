"""
Train ternary ResNet-18 with BINARY ACTIVATIONS (BitNet-style)
Usage: python train/train_binary_ternary.py
"""

import sys
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet_ternary import ternary_resnet18_cifar
from models.ternary_conv import TernaryConv2d


# ───────────────── CONFIG ─────────────────
BATCH_SIZE = 128
EPOCHS = 250
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BEST_PATH = "checkpoints/resnet18_binary_best.pth"
RESUME_PATH = "checkpoints/resnet18_binary_latest.pth"
# ─────────────────────────────────────────


# ───────────────── BINARY ACTIVATION ─────────────────
class BinaryActivation(nn.Module):
    def forward(self, x):
        # scaling factor (important for accuracy)
        alpha = x.abs().mean(dim=(1,2,3), keepdim=True)

        # binarize
        out = torch.sign(x)
        out[out == 0] = 1

        # apply scale
        out = alpha * out

        # STE (straight-through estimator)
        return x + (out - x).detach()
# ────────────────────────────────────────────────────


# ───────────────── PATCH MODEL ─────────────────
def replace_relu_with_binary(model):
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            setattr(model, name, BinaryActivation())
        else:
            replace_relu_with_binary(module)
# ──────────────────────────────────────────────


def get_dataloaders():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    return trainloader, testloader


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        loss_sum += loss.item() * x.size(0)
        _, pred = out.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()

    return loss_sum / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = criterion(out, y)

        loss_sum += loss.item() * x.size(0)
        _, pred = out.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()

    return loss_sum / total, 100.0 * correct / total


def main():
    print("Device:", DEVICE)

    trainloader, testloader = get_dataloaders()

    model = ternary_resnet18_cifar(num_classes=10).to(DEVICE)

    # 🔥 Replace ALL ReLU with BinaryActivation
    replace_relu_with_binary(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    os.makedirs("checkpoints", exist_ok=True)

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer)
        test_loss, test_acc = evaluate(model, testloader, criterion)

        scheduler.step()

        print(f"Epoch {epoch:3d} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), BEST_PATH)

    print("Best Accuracy:", best_acc)


if __name__ == "__main__":
    main()