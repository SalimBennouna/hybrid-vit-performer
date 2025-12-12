from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import TrainConfig


def get_dataloaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader, int]:
    """
    Matches your notebook transforms/params closely.
    """
    if cfg.dataset == "MNIST":
        in_channels = 1
        transform = transforms.Compose([
            transforms.Resize((cfg.img_size, cfg.img_size)),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
        test_ds  = datasets.MNIST("./data", train=False, transform=transform)

    elif cfg.dataset == "CIFAR10":
        in_channels = 3
        transform = transforms.Compose([
            transforms.Resize((cfg.img_size, cfg.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010)
            )
        ])
        train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
        test_ds  = datasets.CIFAR10("./data", train=False, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    use_pin = (cfg.device == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=use_pin
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=use_pin
    )
    return train_loader, test_loader, in_channels