from __future__ import annotations

import functools
import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .config import TrainConfig


class SyntheticImageNet(Dataset):
    """
    Top-level class so it can be pickled with multiprocessing DataLoader.
    Yields random 3xHxW images and random labels in [0, num_classes).
    """
    def __init__(self, length: int, img_size: int, num_classes: int):
        self.length = length
        self.img_size = img_size
        self.num_classes = num_classes

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = torch.randn(3, self.img_size, self.img_size)
        label = torch.randint(low=0, high=self.num_classes, size=(1,)).item()
        return img, label


def _seed_worker(base_seed: int, worker_id: int) -> None:
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def get_dataloaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader, int]:
    """
    Matches your notebook transforms/params closely.
    """

    def make_loader(ds, shuffle: bool) -> DataLoader:
        generator = None
        worker_init_fn = None
        if cfg.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(cfg.seed)
            worker_init_fn = functools.partial(_seed_worker, cfg.seed)

        use_pin = (cfg.device == "cuda")
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=use_pin,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )

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

    elif cfg.dataset == "IMAGENET_SYNTH":
        in_channels = 3

        train_ds = SyntheticImageNet(cfg.synthetic_train_samples, cfg.img_size, cfg.num_classes)
        test_ds = SyntheticImageNet(cfg.synthetic_val_samples, cfg.img_size, cfg.num_classes)

    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    train_loader = make_loader(train_ds, shuffle=True)
    test_loader = make_loader(test_ds, shuffle=False)
    return train_loader, test_loader, in_channels
