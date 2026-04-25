import os
from PIL import Image
from torch.utils.data import Dataset


class CTScanFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Get class names (subfolders)
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        self.class_to_idx = {
            cls: idx for idx, cls in enumerate(self.classes)
        }

        self.samples = []

        # Walk through folders
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)

            for root, _, files in os.walk(class_dir):
                for file in files:
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
                        img_path = os.path.join(root, file)
                        label = self.class_to_idx[class_name]
                        self.samples.append((img_path, label))

        if len(self.samples) == 0:
            raise ValueError(f"No PNG images found in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading {img_path}: {e}")

        if self.transform is not None:
            image = self.transform(image)   # IMPORTANT

        return {
            "image": image,
            "label": label,
            "path": img_path
        }