from torchvision import transforms


def get_train_transforms(image_size: int = 224):
    """
    Training transforms for CT scan images.

    Includes light augmentation suitable for medical data.
    """

    transform = transforms.Compose([
        # Resize to model input size
        transforms.Resize((image_size, image_size)),

        # Light augmentation (safe)
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(p=0.5),

        # Convert to tensor
        transforms.ToTensor(),

        # Normalize (for pretrained models like ResNet/EfficientNet)
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return transform


def get_eval_transforms(image_size: int = 224):
    """
    Validation / Test transforms (NO augmentation).
    """

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return transform