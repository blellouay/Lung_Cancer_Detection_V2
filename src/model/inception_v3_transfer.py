import torch
import torch.nn as nn
import warnings
from torchvision.models import inception_v3, Inception_V3_Weights


class InceptionV3Transfer(nn.Module):
    def __init__(
        self,
        num_classes=4,
        freeze_features=True,
        dropout=0.5,
        aux_logits=False,
        pretrained=True
    ):
        super().__init__()

        weights = Inception_V3_Weights.DEFAULT if pretrained else None
        build_aux_logits = True if weights is not None else aux_logits

        try:
            self.model = inception_v3(
                weights=weights,
                aux_logits=build_aux_logits,
                init_weights=False
            )
        except (OSError, PermissionError, RuntimeError) as e:
            if weights is None:
                raise

            warnings.warn(
                "Could not load pretrained InceptionV3 weights. "
                f"Falling back to random initialization. Original error: {e}",
                RuntimeWarning
            )
            self.model = inception_v3(
                weights=None,
                aux_logits=aux_logits,
                init_weights=False
            )

        if freeze_features:
            for param in self.model.parameters():
                param.requires_grad = False

        # Main classifier
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

        # Auxiliary classifier
        if self.model.aux_logits:
            aux_in_features = self.model.AuxLogits.fc.in_features
            self.model.AuxLogits.fc = nn.Linear(
                aux_in_features,
                num_classes
            )

        if not aux_logits:
            self.model.aux_logits = False
            self.model.AuxLogits = None

    def forward(self, x):
        if self.training and self.model.aux_logits and x.shape[-2:] != (299, 299):
            raise ValueError(
                "InceptionV3Transfer with aux_logits=True must be trained with "
                "299x299 images. Set IMAGE_SIZE = 299 for this model or create "
                "it with aux_logits=False."
            )

        return self.model(x)
