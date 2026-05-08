import torch
import torch.nn as nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_channels):
        super().__init__()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            Swish(),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class MBConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        expand_ratio,
        stride,
        kernel_size,
        se_ratio=0.25,
        drop_rate=0.0
    ):
        super().__init__()

        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []

        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                Swish()
            ])

        layers.extend([
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=hidden_dim,
                bias=False
            ),
            nn.BatchNorm2d(hidden_dim),
            Swish()
        ])

        reduced_channels = max(1, int(in_channels * se_ratio))

        layers.append(
            SqueezeExcitation(hidden_dim, reduced_channels)
        )

        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.block = nn.Sequential(*layers)
        self.dropout = nn.Dropout2d(drop_rate)

    def forward(self, x):
        out = self.block(x)

        if self.use_residual:
            out = self.dropout(out)
            out = out + x

        return out


class EfficientNetB0Scratch(nn.Module):
    def __init__(self, num_classes=4, dropout=0.4):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            Swish()
        )

        self.blocks = nn.Sequential(
            MBConvBlock(32, 16, expand_ratio=1, stride=1, kernel_size=3),

            MBConvBlock(16, 24, expand_ratio=6, stride=2, kernel_size=3),
            MBConvBlock(24, 24, expand_ratio=6, stride=1, kernel_size=3),

            MBConvBlock(24, 40, expand_ratio=6, stride=2, kernel_size=5),
            MBConvBlock(40, 40, expand_ratio=6, stride=1, kernel_size=5),

            MBConvBlock(40, 80, expand_ratio=6, stride=2, kernel_size=3),
            MBConvBlock(80, 80, expand_ratio=6, stride=1, kernel_size=3),
            MBConvBlock(80, 80, expand_ratio=6, stride=1, kernel_size=3),

            MBConvBlock(80, 112, expand_ratio=6, stride=1, kernel_size=5),
            MBConvBlock(112, 112, expand_ratio=6, stride=1, kernel_size=5),

            MBConvBlock(112, 192, expand_ratio=6, stride=2, kernel_size=5),
            MBConvBlock(192, 192, expand_ratio=6, stride=1, kernel_size=5),
            MBConvBlock(192, 192, expand_ratio=6, stride=1, kernel_size=5),

            MBConvBlock(192, 320, expand_ratio=6, stride=1, kernel_size=3)
        )

        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            Swish()
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


def count_parameters(model):
    return {
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
    }