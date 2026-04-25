import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
from collections import Counter


def unnormalize(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img = img * std + mean
    img = img.clamp(0, 1)
    return img


def plot_one_image_per_class(dataset):
    seen_classes = set()
    samples_to_plot = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        label = sample["label"]

        if label not in seen_classes:
            samples_to_plot.append(sample)
            seen_classes.add(label)

        if len(seen_classes) == len(dataset.classes):
            break

    plt.figure(figsize=(4 * len(samples_to_plot), 4))

    for i, sample in enumerate(samples_to_plot):
        img = sample["image"]
        label = sample["label"]

        img = unnormalize(img)
        img = img.permute(1, 2, 0).numpy()

        class_name = dataset.classes[label]

        plt.subplot(1, len(samples_to_plot), i + 1)
        plt.imshow(img)
        plt.title(class_name)
        plt.axis("off")

    plt.show()


def plot_class_distribution(dataset, title="Class Distribution"):
    labels = [label for _, label in dataset.samples]
    counts = Counter(labels)

    class_names = dataset.classes
    values = [counts[i] for i in range(len(class_names))]

    plt.figure(figsize=(8, 5))
    plt.bar(class_names, values)
    
    plt.title(title)
    plt.xlabel("Classes")
    plt.ylabel("Number of samples")
    plt.xticks(rotation=30)
    
    # Show values on top of bars (important)
    for i, v in enumerate(values):
        plt.text(i, v + 2, str(v), ha='center')

    plt.show()


def plot_class_distribution(dataset, title="Class Distribution"):
    labels = [label for _, label in dataset.samples]
    counts = Counter(labels)

    class_names = dataset.classes
    values = [counts[i] for i in range(len(class_names))]

    plt.figure(figsize=(8, 5))
    plt.bar(class_names, values)
    
    plt.title(title)
    plt.xlabel("Classes")
    plt.ylabel("Number of samples")
    plt.xticks(rotation=30)
    
    # Show values on top of bars (important)
    for i, v in enumerate(values):
        plt.text(i, v + 2, str(v), ha='center')

    plt.show()