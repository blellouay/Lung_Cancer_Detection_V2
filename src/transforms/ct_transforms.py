from torchvision import transforms


def get_train_transforms(image_size: int = 224):
    """
    Training transforms for CT scan images.

    Safe medical augmentation:
    - light rotation
    - light translation / scale
    - optional horizontal flip
    - no color distortion
    - no random erase
    """

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),

        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=7,
                translate=(0.03, 0.03),
                scale=(0.95, 1.05)
            )
        ], p=0.5),

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_eval_transforms(image_size: int = 224):
    """
    Validation / test transforms.

    No augmentation here.
    """

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])