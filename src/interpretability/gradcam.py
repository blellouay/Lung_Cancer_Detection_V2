import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.utils.model_outputs import get_main_logits


def denormalize_image(tensor_image, mean, std):
    image = tensor_image.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))

    mean = np.array(mean)
    std = np.array(std)

    image = image * std + mean
    image = np.clip(image, 0, 1)

    return image


def get_target_layer(model):
    """
    Automatically select the last Conv2d layer.
    Works with CNNBaseline, ResNet, DenseNet, EfficientNet, custom CNNs, etc.
    """

    conv_layers = []

    for name, module in model.named_modules():
        if "AuxLogits" in name:
            continue

        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))

    if len(conv_layers) == 0:
        raise ValueError(
            "No Conv2d layer found. Grad-CAM requires a CNN-like model."
        )

    last_conv_name, last_conv_layer = conv_layers[-1]

    print(f"Grad-CAM target layer selected: {last_conv_name}")

    return last_conv_layer


def generate_gradcam(
    model,
    image_tensor,
    target_class,
    save_path,
    device,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
):
    model.eval()
    model.to(device)

    input_tensor = image_tensor.unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    target_layers = [get_target_layer(model)]
    targets = [ClassifierOutputTarget(target_class)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )[0]

    rgb_image = denormalize_image(image_tensor, mean, std)

    cam_overlay = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 8))
    plt.imshow(cam_overlay)
    plt.axis("off")
    plt.title(f"Grad-CAM - class {target_class}")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

    return save_path


def generate_gradcam_samples(
    model,
    dataloader,
    device,
    save_dir,
    class_names=None,
    num_images=8,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
):
    os.makedirs(save_dir, exist_ok=True)

    saved_paths = []
    count = 0

    model.eval()
    model.to(device)

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"]

        with torch.no_grad():
            outputs = model(images)
            preds = torch.argmax(get_main_logits(outputs), dim=1)

        for i in range(images.size(0)):
            if count >= num_images:
                return saved_paths

            image_tensor = images[i].detach().cpu()
            pred_class = preds[i].item()
            true_class = labels[i].item()

            pred_name = class_names[pred_class] if class_names else str(pred_class)
            true_name = class_names[true_class] if class_names else str(true_class)

            filename = f"gradcam_{count}_true_{true_name}_pred_{pred_name}.png"
            save_path = os.path.join(save_dir, filename)

            generate_gradcam(
                model=model,
                image_tensor=image_tensor,
                target_class=pred_class,
                save_path=save_path,
                device=device,
                mean=mean,
                std=std
            )

            saved_paths.append(save_path)
            count += 1

    return saved_paths
