import os
import json
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms


def get_cifar10_dataloaders(config):
    """
    Create deterministic CIFAR-10 dataloaders with train/val split

    Returns: train_loader, val_loader, test_loader
    """
    data_dir = config["data_dir"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    train_val_split = config["train_val_split"]

    # CIFAR-10 normalization values
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    # Data transforms
    train_transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)

    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    # Create deterministic train/val split
    total_train = len(train_dataset)
    train_size = int(train_val_split * total_train)
    val_size = total_train - train_size

    # Use manual seed for deterministic split
    torch.manual_seed(config["seed"])
    train_indices, val_indices = torch.utils.data.random_split(range(total_train), [train_size, val_size])
    train_indices = train_indices.indices
    val_indices = val_indices.indices

    # Save split indices for reproducibility
    split_info = {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "seed": config["seed"],
    }

    split_path = os.path.join(config["log_dir"], "data_split.json")
    save_split_indices(split_info, split_path)

    # Create subsets
    train_subset = Subset(train_dataset, train_indices)

    # Create validation dataset with test transforms (no augmentation)
    val_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=test_transform)
    val_subset = Subset(val_dataset, val_indices)

    # Create dataloaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"Dataset loaded: {len(train_subset)} train, {len(val_subset)} val, {len(test_dataset)} test")
    print(f"Split indices saved to: {split_path}")

    return train_loader, val_loader, test_loader


def save_split_indices(split_info, save_path):
    """Save split indices for reproducibility"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"Train/val split: {split_info['train_size']}/{split_info['val_size']} (seed: {split_info['seed']})")
