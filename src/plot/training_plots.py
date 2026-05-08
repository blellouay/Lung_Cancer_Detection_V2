import os
import json
import re
import shutil
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULT_DIR_ALIASES = {
    "results/ResNet": "results/Resnet",
    "results/EddicientNet": "results/EfficientNet",
    "results/EffcientNet": "results/EfficientNet",
    "results/ResNet_transf": "results/ResNet_transfer",
    "results/Resnet_transf": "results/ResNet_transfer",
    "results/Reset_transf": "results/ResNet_transfer",
    "results/Inception_tranfer": "results/Inception_transfer",
    "results/inception_transfer": "results/Inception_transfer",
}


def normalize_results_dir(save_dir):
    normalized = str(save_dir).replace("\\", "/").rstrip("/")

    return RESULT_DIR_ALIASES.get(normalized, normalized)


def _move_existing_artifact(old_dir, new_dir, filename):
    old_path = os.path.join(old_dir, filename)
    new_path = os.path.join(new_dir, filename)

    if os.path.abspath(old_path) == os.path.abspath(new_path):
        return

    if os.path.exists(old_path) and not os.path.exists(new_path):
        try:
            shutil.move(old_path, new_path)
        except Exception as e:
            print(f"Could not move {old_path} to {new_path}: {e}")


def _clean_history(history):
    cleaned = {}

    for key, values in history.items():
        cleaned[key] = [
            float(value) if isinstance(value, (int, float)) else value
            for value in values
        ]

    return cleaned


def _safe_filename(name):
    return re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower()).strip("_")


def _plot_curve(history, train_key, valid_key, ylabel, title, save_path):
    if train_key not in history or valid_key not in history:
        return None

    if not history[train_key] or not history[valid_key]:
        return None

    plt.figure(figsize=(8, 6))
    plt.plot(history[train_key], label=train_key.replace("_", " ").title())
    plt.plot(history[valid_key], label=valid_key.replace("_", " ").title())
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return save_path


def save_training_history(history, save_dir):
    original_save_dir = save_dir
    save_dir = normalize_results_dir(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    history_path = os.path.join(save_dir, "training_history.json")

    with open(history_path, "w") as f:
        json.dump(_clean_history(history), f, indent=4)

    if str(original_save_dir) != save_dir:
        _move_existing_artifact(original_save_dir, save_dir, "training_history.json")

    print(f"Training history saved to: {history_path}")


def save_training_curves(history, save_dir):
    original_save_dir = save_dir
    save_dir = normalize_results_dir(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    curve_specs = [
        (
            "train_loss",
            "valid_loss",
            "Loss",
            "Training vs Validation Loss",
            "loss_curve.png",
        ),
        (
            "train_acc",
            "valid_acc",
            "Accuracy",
            "Training vs Validation Accuracy",
            "accuracy_curve.png",
        ),
        (
            "finetune_train_loss",
            "finetune_valid_loss",
            "Loss",
            "Fine-Tuning Train vs Validation Loss",
            "finetune_loss_curve.png",
        ),
        (
            "finetune_train_acc",
            "finetune_valid_acc",
            "Accuracy",
            "Fine-Tuning Train vs Validation Accuracy",
            "finetune_accuracy_curve.png",
        ),
    ]

    saved_paths = []
    for train_key, valid_key, ylabel, title, filename in curve_specs:
        saved_paths.append(
            _plot_curve(
                history,
                train_key,
                valid_key,
                ylabel,
                title,
                os.path.join(save_dir, filename),
            )
        )

    custom_metrics = [
        key[6:]
        for key in history
        if key.startswith("train_")
        and f"valid_{key[6:]}" in history
        and key not in {"train_loss", "train_acc"}
        and not key.startswith("finetune_")
    ]

    for metric_name in custom_metrics:
        filename = f"{_safe_filename(metric_name)}_curve.png"
        saved_paths.append(
            _plot_curve(
                history,
                f"train_{metric_name}",
                f"valid_{metric_name}",
                metric_name.replace("_", " ").title(),
                f"Training vs Validation {metric_name.replace('_', ' ').title()}",
                os.path.join(save_dir, filename),
            )
        )

    for path in saved_paths:
        if path:
            print(f"Training curve saved to: {path}")

    if str(original_save_dir) != save_dir:
        for _, _, _, _, filename in curve_specs:
            _move_existing_artifact(original_save_dir, save_dir, filename)
