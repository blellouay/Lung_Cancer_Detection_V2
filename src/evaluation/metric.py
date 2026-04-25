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
                color="white" if value > cm_display.max() / 2 else "black"
            )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.show()


def compute_metrics(y_true, y_pred, class_names, plot_cm=True):
    cm = confusion_matrix(y_true, y_pred)

    if plot_cm:
        plot_confusion_matrix(cm, class_names, normalize=True)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro_sensitivity": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": cm,
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0
        )
    }