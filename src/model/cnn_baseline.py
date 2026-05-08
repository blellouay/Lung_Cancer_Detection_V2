import torch
import torch.nn as nn


class CNNBaseline(nn.Module):
    def __init__(self, num_classes=4, dropout=0.5):
        super(CNNBaseline, self).__init__()

        self.features = nn.Sequential(
            # =========================
            # Block 1
            # =========================
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # =========================
            # Block 2
            # =========================
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # =========================
            # Block 3
            # =========================
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Adaptive pooling makes architecture more flexible
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                128 * 4 * 4,
                256
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(
                256,
                128
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


def count_parameters(model):
    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params
    }