# src/deployment/model_loader.py

import torch


def get_model_registry():
    """
    Add every model architecture here.
    The key must match model.__class__.__name__ saved in metrics.json.
    """

    from src.model.cnn_baseline import CNNBaseline
    from src.model.resnet import ResNet18Scratch
    from src.model.vgg16 import VGG16Scratch
    from src.model.mobilenetv2_scratch import MobileNetV2Scratch
    from src.model.efficientnet_b0 import EfficientNetB0Scratch
    from src.model.resnet18_transfer import ResNet18Transfer
    from src.model.inception_v3_transfer import InceptionV3Transfer
    from src.model.efficientnetb3 import EfficientNetB3

    return {
        "CNNBaseline": CNNBaseline,
        "ResNetScratch": ResNet18Scratch,
        "ResNet18Scratch": ResNet18Scratch,
        "ResNet18": ResNet18Scratch,
        "ResNet18Transfer": ResNet18Transfer,
        "ResNetTransfer": ResNet18Transfer,
        "VGG16Scratch": VGG16Scratch,
        "VGG16": VGG16Scratch,
        "MobileNetV2Scratch": MobileNetV2Scratch,
        "EfficientNetB0Scratch": EfficientNetB0Scratch,
        "EfficientNetB0": EfficientNetB0Scratch,
        "EfficientNet": EfficientNetB0Scratch,
        "InceptionV3Transfer": InceptionV3Transfer,
        "InceptionV3": InceptionV3Transfer,
        "Inception": InceptionV3Transfer,
        "EfficientNetB3Transfer": EfficientNetB3
    }


def build_model(model_name: str, num_classes: int, model_kwargs: dict = None):
    registry = get_model_registry()
    model_name = (model_name or "").strip()

    if model_name not in registry:
        available = list(registry.keys())
        raise ValueError(
            f"Unknown model architecture: {model_name}. "
            f"Available models: {available}"
        )

    model_kwargs = model_kwargs or {}

    model_class = registry[model_name]
    if model_class.__name__ in {"ResNet18Transfer", "InceptionV3Transfer", "EfficientNetB3"}:
        model_kwargs.setdefault("pretrained", False)

    return model_class(
        num_classes=num_classes,
        **model_kwargs
    )


def load_trained_model(
    model_name: str,
    model_path: str,
    num_classes: int,
    device,
    model_kwargs: dict = None
):
    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        model_kwargs=model_kwargs
    )

    state_dict = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model
