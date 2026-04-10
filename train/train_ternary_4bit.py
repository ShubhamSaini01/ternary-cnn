"""
Train ternary ResNet-18 with 4-bit activation quantization-aware training (QAT).

Ternary weights {-1, 0, +1} + 4-bit activations [0..15] post-ReLU.
Uses STE for both weight and activation quantization.

Usage:
    python train/train_ternary_4bit.py

Saves:
    checkpoints/resnet18_ternary_4bit_best.pth
    checkpoints/resnet18_ternary_4bit_latest.pth (for resume)
"""

import sys, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── 4-bit activation quantizer (STE) ───────────────────────

class Quantize4bit(torch.autograd.Function):
    """Quantize post-ReLU activations to 4-bit (16 levels).
    Forward: round to [0, 15] * scale
    Backward: STE (pass gradients straight through)
    """
    @staticmethod
    def forward(ctx, x):
        # x is post-ReLU, so x >= 0
        max_val = x.detach().max()
        if max_val < 1e-10:
            return x
        scale = max_val / 15.0
        # Quantize: round(x / scale) * scale
        x_q = torch.round(x / scale).clamp(0, 15) * scale
        return x_q

    @staticmethod
    def backward(ctx, grad_output):
        # STE: pass gradients through
        return grad_output

def quantize_4bit(x):
    return Quantize4bit.apply(x)

# ─── Ternary quantizer (same as existing) ───────────────────

class TernaryQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights):
        abs_weights = weights.abs()
        delta = 0.7 * abs_weights.mean(dim=(1, 2, 3), keepdim=True)
        ternary = torch.zeros_like(weights)
        ternary[weights > delta] = 1.0
        ternary[weights < -delta] = -1.0
        mask = ternary != 0
        count = mask.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1)
        alpha = (abs_weights * mask.float()).sum(dim=(1, 2, 3), keepdim=True) / count
        ctx.save_for_backward(weights, delta)
        return ternary, alpha

    @staticmethod
    def backward(ctx, grad_ternary, grad_alpha):
        weights, delta = ctx.saved_tensors
        grad = grad_ternary.clone()
        grad[weights.abs() > 1.0] = 0
        return grad

class TernaryConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        ternary_w, alpha = TernaryQuantize.apply(self.weight)
        scaled_w = alpha * ternary_w
        return F.conv2d(x, scaled_w, self.bias, stride=self.stride, padding=self.padding)

# ─── ResNet-18 with ternary weights + 4-bit activations ─────

class BasicBlock4bit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = TernaryConv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = TernaryConv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = quantize_4bit(F.relu(self.bn1(self.conv1(x))))  # 4-bit after ReLU
        out = self.bn2(self.conv2(out))
        out = quantize_4bit(F.relu(out + self.shortcut(x)))    # 4-bit after add+ReLU
        return out

class TernaryResNet18_4bit(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64
        # First conv stays FP32
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock4bit(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = quantize_4bit(F.relu(self.bn1(self.conv1(x))))  # 4-bit after first ReLU
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)

# ─── Config ──────────────────────────────────────────────────
BATCH_SIZE = 128
EPOCHS = 200
LR = 0.05
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BEST_PATH = "checkpoints/resnet18_ternary_4bit_best.pth"
RESUME_PATH = "checkpoints/resnet18_ternary_4bit_latest.pth"

# ─── Data ────────────────────────────────────────────────────
def get_dataloaders():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    return trainloader, testloader

# ─── Training ────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = correct = total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
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
    running_loss = correct = total = 0
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
    print(f"Training: Ternary weights + 4-bit activations QAT")
    print()

    trainloader, testloader = get_dataloaders()
    model = TernaryResNet18_4bit(num_classes=10).to(DEVICE)

    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    ternary_params = sum(p.numel() for n, p in model.named_parameters()
                         if any(f'layer{i}' in n and 'conv' in n and 'weight' in n
                                for i in range(1,5)))
    print(f"Total parameters: {total_params:,}")
    print(f"Ternary params: {ternary_params:,}")
    print()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    os.makedirs("checkpoints", exist_ok=True)

    # Resume
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
        print(f"Resumed at epoch {start_epoch}, best acc: {best_acc:.2f}%\n")

    start_time = time.time()
    for epoch in range(start_epoch, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, DEVICE)
        test_loss, test_acc = evaluate(model, testloader, criterion, DEVICE)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{EPOCHS} | LR: {lr:.4f} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'test_acc': test_acc,
            }, BEST_PATH)
            print(f"  -> Saved best model (acc: {best_acc:.2f}%)")

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
