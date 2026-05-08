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

    return {
        "CNNBaseline": CNNBaseline,
        "ResNetScratch": ResNet18Scratch,
        "ResNet18Scratch": ResNet18Scratch,
        "ResNet18": ResNet18Scratch,
        "VGG16Scratch": VGG16Scratch,
        "VGG16": VGG16Scratch,
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
