from torch.utils.data import DataLoader

from .data import CTScanFolderDataset

from src.transforms.ct_transforms import (
    get_train_transforms,
    get_eval_transforms
)


def create_dataloaders(
    train_dir,
    valid_dir,
    test_dir,
    image_size=224,
    batch_size=32,
    num_workers=2
):
    """
    Create train, validation, and test DataLoaders.
    """

    train_dataset = CTScanFolderDataset(
        root_dir=train_dir,
        transform=get_train_transforms(image_size)
    )

    valid_dataset = CTScanFolderDataset(
        root_dir=valid_dir,
        transform=get_eval_transforms(image_size)
    )

    test_dataset = CTScanFolderDataset(
        root_dir=test_dir,
        transform=get_eval_transforms(image_size)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, valid_loader, test_loader, train_dataset.classes