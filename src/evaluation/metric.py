# src/evaluation/metric.py

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def plot_confusion_matrix(cm, class_names, normalize=True):
    """
    Plot confusion matrix.

    Args:
        cm (np.ndarray): confusion matrix
        class_names (list): list of class labels
        normalize (bool): whether to normalize values
    """

    cm = np.array(cm)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)

        cm_display = np.divide(
            cm.astype(float),
            row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0
        )
    else:
        cm_display = cm

    plt.figure(figsize=(9, 7))
    plt.imshow(cm_display, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)

    threshold = cm_display.max() / 2

    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            value = cm_display[i, j]

            text = f"{value:.2f}" if normalize else str(int(value))

            plt.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else "black"
            )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()


def calculate_classification_metrics(y_true, y_pred):
    """
    Basic weighted classification metrics.
    Useful for quick evaluation.
    """

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "f1_score": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
    }


def compute_metrics(y_true, y_pred, class_names, plot_cm=True):
    """
    Full evaluation metrics for project experiments.
    Includes:
    - accuracy
    - macro precision
    - macro recall
    - macro f1
    - confusion matrix
    - classification report
    """

    cm = confusion_matrix(y_true, y_pred)

    if plot_cm:
        plot_confusion_matrix(
            cm=cm,
            class_names=class_names,
            normalize=True
        )

    return {
        "accuracy": accuracy_score(y_true, y_pred),

        "precision_macro": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "recall_macro": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "confusion_matrix": cm.tolist(),

        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0
        )
    }