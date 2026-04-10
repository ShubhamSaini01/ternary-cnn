"""
ResNet-18 with ternary convolutions for CIFAR-10.

Design decisions:
- First conv and shortcut projections stay full-precision (standard practice)
  → First layer captures low-level features, too important to quantize
  → Shortcut 1x1 convs have few params, quantizing them hurts residual flow
- All 3x3 convs in BasicBlocks are ternary (this is where the bulk of params are)
- Final FC layer stays full-precision (negligible param count)
- BatchNorm is critical — stabilizes activations after ternary convolutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
#from ternary_conv import TernaryConv2d
from models.ternary_conv import TernaryConv2d

class TernaryBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Ternary 3x3 convolutions (the main workhorses)
        self.conv1 = TernaryConv2d(in_channels, out_channels, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = TernaryConv2d(out_channels, out_channels, kernel_size=3,
                                   stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut stays full-precision
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class TernaryResNet18CIFAR(nn.Module):
    """ResNet-18 with ternary convolutions, adapted for CIFAR-10."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64

        # First conv stays full-precision (standard practice)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # ResNet blocks with ternary convolutions
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # FC stays full-precision
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(TernaryBasicBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ternary_resnet18_cifar(num_classes=10):
    return TernaryResNet18CIFAR(num_classes=num_classes)
