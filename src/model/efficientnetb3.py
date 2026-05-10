import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B3_Weights
 
 
class EfficientNetB3(nn.Module):
    """
    EfficientNet-B3 classifier for CT scan images.
 
    Args:
        num_classes: Number of output classes.
        pretrained:  Load ImageNet weights if True.
        dropout:     Dropout rate before the final linear layer.
    """
 
    def __init__(self, num_classes: int, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
 
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b3(weights=weights)
 
        # Keep everything except the original classifier
        self.features = backbone.features
        self.avgpool  = backbone.avgpool
 
        in_features = backbone.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )
 
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

