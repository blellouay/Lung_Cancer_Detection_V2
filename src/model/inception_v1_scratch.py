# src/model/googlenet_scratch.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------
# Convenience: Conv + BN + ReLU block
# --------------------------------------------------
class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# --------------------------------------------------
# Inception Module (Inception v1 style)
#
# Four parallel branches:
#   1 × 1 conv
#   1 × 1 conv  →  3 × 3 conv
#   1 × 1 conv  →  5 × 5 conv
#   3 × 3 max-pool  →  1 × 1 conv
#
# Outputs are depth-concatenated along dim=1.
# --------------------------------------------------
class InceptionModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_1x1,          # branch 1
        red_3x3, out_3x3, # branch 2: reduction then 3×3
        red_5x5, out_5x5, # branch 3: reduction then 5×5
        out_pool          # branch 4: pool projection
    ):
        super().__init__()

        # Branch 1 — 1×1 conv
        self.branch1 = ConvBnRelu(in_channels, out_1x1, kernel_size=1)

        # Branch 2 — 1×1 reduction → 3×3 conv
        self.branch2 = nn.Sequential(
            ConvBnRelu(in_channels, red_3x3, kernel_size=1),
            ConvBnRelu(red_3x3, out_3x3, kernel_size=3, padding=1)
        )

        # Branch 3 — 1×1 reduction → 5×5 conv
        self.branch3 = nn.Sequential(
            ConvBnRelu(in_channels, red_5x5, kernel_size=1),
            ConvBnRelu(red_5x5, out_5x5, kernel_size=5, padding=2)
        )

        # Branch 4 — 3×3 max-pool → 1×1 projection
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBnRelu(in_channels, out_pool, kernel_size=1)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        return torch.cat([b1, b2, b3, b4], dim=1)


# --------------------------------------------------
# Auxiliary Classifier (used during training only)
# Provides gradient signal to earlier layers.
# --------------------------------------------------
class AuxClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        self.conv = ConvBnRelu(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(128 * 4 * 4, 1024)

        self.fc2 = nn.Linear(1024, num_classes)

        self.dropout = nn.Dropout(p=0.7)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.avgpool(x)

        x = self.conv(x)

        x = torch.flatten(x, 1)

        x = self.relu(self.fc1(x))

        x = self.dropout(x)

        x = self.fc2(x)

        return x


# --------------------------------------------------
# Full GoogLeNet (Inception v1) Architecture
# --------------------------------------------------
class GoogLeNetScratch(nn.Module):
    def __init__(self, num_classes=4, use_aux=True):
        super().__init__()

        self.use_aux = use_aux

        # ── Stem ──────────────────────────────────
        self.conv1 = ConvBnRelu(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = ConvBnRelu(64, 64, kernel_size=1)
        self.conv3 = ConvBnRelu(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ── Inception 3a / 3b ─────────────────────
        #                        1x1  r3  3x3  r5  5x5  pool
        self.inception3a = InceptionModule(192,  64,  96, 128, 16,  32,  32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32,  96,  64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ── Inception 4a–4e ───────────────────────
        self.inception4a = InceptionModule(480, 192,  96, 208, 16,  48,  64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24,  64,  64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24,  64,  64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32,  64,  64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ── Inception 5a / 5b ─────────────────────
        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        # ── Auxiliary classifiers (after 4a and 4d) ─
        if self.use_aux:
            self.aux1 = AuxClassifier(512, num_classes)
            self.aux2 = AuxClassifier(528, num_classes)

        # ── Classification head ───────────────────
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(p=0.4)

        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        # Inception 3
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        # Inception 4
        x = self.inception4a(x)

        aux1_out = None
        if self.use_aux and self.training:
            aux1_out = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        aux2_out = None
        if self.use_aux and self.training:
            aux2_out = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)

        # Inception 5
        x = self.inception5a(x)
        x = self.inception5b(x)

        # Head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.use_aux and self.training:
            return x, aux1_out, aux2_out

        return x

    # 🔹 Total parameters
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    # 🔹 Trainable parameters only
    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --------------------------------------------------
# GoogLeNet Factory
# --------------------------------------------------
def GoogLeNetFromScratch(num_classes=4, use_aux=True):
    return GoogLeNetScratch(
        num_classes=num_classes,
        use_aux=use_aux
    )