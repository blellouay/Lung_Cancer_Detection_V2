import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Transfer(nn.Module):
    def __init__(self, num_classes=4, freeze_features=True, pretrained=True):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.model = resnet18(weights=weights)

        if freeze_features:
            for param in self.model.parameters():
                param.requires_grad = False

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
