"""
Train ternary ResNet-18 (TWN) on CIFAR-10.
Usage: python train/train_ternary.py

Resume support: if training crashes, just rerun — it picks up from the last epoch.

Training notes vs baseline:
- Lower initial LR (0.05 vs 0.1) — ternary weights are sensitive to large updates
- Longer training possible if needed (200 epochs should be fine)
- Same augmentation and optimizer as baseline for fair comparison
"""

import sys
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.resnet_ternary import ternary_resnet18_cifar
from models.ternary_conv import TernaryConv2d

# ─── Config ───────────────────────────────────────────────────
BATCH_SIZE = 128
EPOCHS = 200
LR = 0.05          # Lower than FP baseline — ternary is sensitive
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BEST_PATH = "checkpoints/resnet18_ternary_best.pth"
RESUME_PATH = "checkpoints/resnet18_ternary_latest.pth"
# ──────────────────────────────────────────────────────────────


def get_dataloaders():
    """CIFAR-10 with standard data augmentation (same as baseline)."""
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

    trainset = datasets.CIFAR10(root='./data', train=True,
                                download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False,
                               download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True)

    return trainloader, testloader


def count_params(model):
    """Count total and ternary parameters."""
    total = 0
    ternary = 0
    for name, param in model.named_parameters():
        total += param.numel()
        # TernaryConv2d layers have 'weight' in their name within ternary blocks
        # Check if the parent module is a TernaryConv2d
        if param.requires_grad:
            # Walk the module tree to find ternary conv weights
            parts = name.split('.')
            module = model
            for part in parts[:-1]:
                module = getattr(module, part)
            from models.ternary_conv import TernaryConv2d
            if isinstance(module, TernaryConv2d) and parts[-1] == 'weight':
                ternary += param.numel()
    return total, ternary


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # Gradient clipping — helps stabilize STE training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / total, 100.0 * correct / total


def main():
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Data
    trainloader, testloader = get_dataloaders()
    print(f"Training samples: {len(trainloader.dataset)}")
    print(f"Test samples: {len(testloader.dataset)}")

    # Model
    model = ternary_resnet18_cifar(num_classes=10).to(DEVICE)
    total_params, ternary_params = count_params(model)
    fp_params = total_params - ternary_params
    print(f"Total parameters: {total_params:,}")
    print(f"  Ternary params: {ternary_params:,} ({100*ternary_params/total_params:.1f}%)")
    print(f"  FP32 params:    {fp_params:,} ({100*fp_params/total_params:.1f}%)")
    print(f"Model size (if all ternary): {ternary_params * 2 / 8 / 1e6:.2f} MB (ternary) + {fp_params * 4 / 1e6:.2f} MB (FP32)")
    print()

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Create checkpoint dir
    os.makedirs("checkpoints", exist_ok=True)

    # ─── Resume from checkpoint if it exists ───
    start_epoch = 1
    best_acc = 0.0
    if os.path.exists(RESUME_PATH):
        print(f"Resuming from {RESUME_PATH}...")
        ckpt = torch.load(RESUME_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt['best_acc']
        print(f"Resumed at epoch {start_epoch}, best acc so far: {best_acc:.2f}%")
        print()

    # Training loop
    start_time = time.time()

    for epoch in range(start_epoch, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, trainloader, criterion, optimizer, DEVICE)
        test_loss, test_acc = evaluate(
            model, testloader, criterion, DEVICE)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"LR: {lr:.4f} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}%")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'test_acc': test_acc,
                'test_loss': test_loss,
            }, BEST_PATH)
            print(f"  ✓ Saved best model (acc: {best_acc:.2f}%)")

        # Always save latest for crash recovery
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
        }, RESUME_PATH)

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed/60:.1f} minutes")
    print(f"Best test accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
