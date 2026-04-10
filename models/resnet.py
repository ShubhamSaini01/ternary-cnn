"""ResNet-18 for CIFAR-10 with optional ternary convolutions.

CIFAR-10 variant: uses 3x3 initial conv (no maxpool) since input is 32x32.
"""

import torch.nn as nn
from models.ternary_conv import TernaryConv2d


def _conv3x3(in_c, out_c, stride=1, ternary=False):
    cls = TernaryConv2d if ternary else nn.Conv2d
    return cls(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)


def _conv1x1(in_c, out_c, stride=1, ternary=False):
    cls = TernaryConv2d if ternary else nn.Conv2d
    return cls(in_c, out_c, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, ternary=False):
        super().__init__()
        self.conv1 = _conv3x3(in_planes, planes, stride, ternary=ternary)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes, ternary=ternary)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    """ResNet-18 adapted for CIFAR-10 (32x32 input)."""

    def __init__(self, num_classes=10, ternary=False):
        super().__init__()
        self.ternary = ternary
        self.in_planes = 64

        # CIFAR-10: small initial conv, no maxpool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                _conv1x1(self.in_planes, planes, stride, ternary=self.ternary),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicBlock(self.in_planes, planes, stride, downsample, ternary=self.ternary)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes, ternary=self.ternary))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
