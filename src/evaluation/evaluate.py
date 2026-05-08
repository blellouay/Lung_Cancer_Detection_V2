import torch
import os
from tqdm import tqdm
from sklearn.metrics import recall_score

from .metric import compute_metrics
from src.utils.model_outputs import get_main_logits


def evaluate_model(
    model,
    data_loader,
    device,
    class_names,
    plot_cm=True,
    generate_gradcam=False,
    gradcam_save_dir=None,
    results_dir=None,
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
            preds = torch.argmax(get_main_logits(outputs), dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(
        y_true=all_labels,
        y_pred=all_preds,
        class_names=class_names,
        plot_cm=plot_cm
    )

    # =========================
    # Per-class recall
    # =========================
    per_class_recall_values = recall_score(
        all_labels,
        all_preds,
        average=None,
        labels=list(range(len(class_names))),
        zero_division=0
    )

    per_class_recall = {
        class_names[i]: float(per_class_recall_values[i])
        for i in range(len(class_names))
    }

    metrics["per_class_recall"] = per_class_recall
    metrics["min_per_class_recall"] = float(per_class_recall_values.min())
    metrics["weakest_class"] = min(
        per_class_recall,
        key=per_class_recall.get
    )

    if generate_gradcam:
        from src.interpretability.gradcam import generate_gradcam_samples

        if gradcam_save_dir is None:
            gradcam_save_dir = (
                os.path.join(results_dir, "gradcam")
                if results_dir is not None
                else "results/gradcam"
            )

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
