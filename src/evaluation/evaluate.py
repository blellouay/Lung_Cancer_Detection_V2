import os
import torch
from tqdm import tqdm

from .metric import compute_metrics


def evaluate_model(
    model,
    data_loader,
    device,
    class_names,
    plot_cm=True,
    generate_gradcam=False,
    gradcam_save_dir=None,
    num_gradcam_images=8
):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(
        y_true=all_labels,
        y_pred=all_preds,
        class_names=class_names,
        plot_cm=plot_cm
    )

    if generate_gradcam:
        from src.interpretability.gradcam import generate_gradcam_samples

        if gradcam_save_dir is None:
            gradcam_save_dir = "results/gradcam"

        gradcam_paths = generate_gradcam_samples(
            model=model,
            dataloader=data_loader,
            device=device,
            save_dir=gradcam_save_dir,
            class_names=class_names,
            num_images=num_gradcam_images
        )

        metrics["gradcam_paths"] = gradcam_paths

    return metrics