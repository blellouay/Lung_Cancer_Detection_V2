import torch
from tqdm import tqdm

from .metric import compute_metrics


def evaluate_model(model, data_loader, device, class_names):
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
        class_names=class_names
    )

    return metrics