"""Train the Potato vs Onion image classifier (2-Class Setup).

Expected dataset layout (ImageFolder style)::

    dataset/
        onions/     *.jpg
        potatoes/   *.jpg

Example:
    python train_model.py --data-dir ./dataset --model mobilenet_v2 --epochs 25
    python train_model.py --model cnn --img-size 128 --epochs 40
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split
from torchvision import datasets, transforms

from model import build_model, count_parameters

logger = logging.getLogger("train")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Setup helpers
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Train the Potato/Onion classifier.")
    p.add_argument("--data-dir", type=Path, default=Path("./dataset"))
    p.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    p.add_argument("--model", choices=["cnn", "mobilenet_v2"], default="mobilenet_v2")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--freeze-epochs",
        type=int,
        default=0,
        help="(mobilenet_v2 only) train only the new head for this many epochs.",
    )
    p.add_argument("--no-pretrained", action="store_true",
                   help="Do not load ImageNet weights for mobilenet_v2.")
    return p.parse_args()


def setup_logging(output_dir: Path) -> None:
    """Log to both the console and ``output_dir/train.log``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "train.log", mode="w"),
        ],
    )


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducible splits and training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CUDA if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
    else:
        logger.info("CUDA not available, using CPU.")
    return device


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def build_transforms(img_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    """Create (train, validation) transform pipelines.

    Training uses random augmentation; validation uses only deterministic
    resizing and normalization so metrics are comparable between epochs.
    """
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def build_sampler(
    train_view: datasets.ImageFolder,
    train_indices: list[int],
    num_classes: int,
    seed: int,
) -> WeightedRandomSampler:
    """Create a ``WeightedRandomSampler`` with inverse-class-frequency weights."""
    train_targets = [train_view.targets[i] for i in train_indices]
    counts = Counter(train_targets)
    class_weights = {c: 1.0 / counts[c] for c in counts}
    sample_weights = torch.tensor([class_weights[t] for t in train_targets], dtype=torch.double)

    logger.info(
        "Train class counts: %s",
        {train_view.classes[c]: counts.get(c, 0) for c in range(num_classes)},
    )
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, list[str]]:
    """Split the dataset 80/20 and build data loaders for 2 classes."""
    train_tf, val_tf = build_transforms(args.img_size)

    # Load dataset to inspect classes dynamically
    val_view = datasets.ImageFolder(args.data_dir, transform=val_tf)
    classes = val_view.classes
    logger.info("Classes detected: %s", classes)
    
    if len(classes) < 2:
        raise ValueError(f"Expected at least 2 classes in '{args.data_dir}', found {len(classes)}: {classes}")

    train_view = datasets.ImageFolder(args.data_dir, transform=train_tf)

    n_total = len(train_view)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val
    split = random_split(
        range(n_total),
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_indices = list(split[0])
    train_set = Subset(train_view, train_indices)
    val_set = Subset(val_view, list(split[1]))
    logger.info("Dataset: %d images -> %d train / %d validation", n_total, n_train, n_val)

    sampler = build_sampler(train_view, train_indices, len(classes), args.seed)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)
    return train_loader, val_loader, classes


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one pass over ``loader``."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += inputs.size(0)

    return total_loss / total, 100.0 * correct / total


def main() -> None:
    """Entry point: build data/model, train, checkpoint the best model."""
    args = parse_args()
    setup_logging(args.output_dir)
    set_seed(args.seed)
    device = get_device()

    train_loader, val_loader, classes = build_dataloaders(args)

    model_kwargs: dict = {}
    if args.model == "mobilenet_v2":
        model_kwargs = {
            "pretrained": not args.no_pretrained,
            "freeze_backbone": args.freeze_epochs > 0,
        }
    model = build_model(args.model, num_classes=len(classes), **model_kwargs).to(device)
    logger.info("Model: %s | trainable parameters: %s", args.model, f"{count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_path = args.output_dir / "best_model.pth"
    (args.output_dir / "classes.json").write_text(json.dumps(classes))
    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
    }
    best_val_acc = 0.0

    logger.info("Starting training for %d epochs...", args.epochs)
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        if args.model == "mobilenet_v2" and args.freeze_epochs and epoch == args.freeze_epochs + 1:
            model.unfreeze_backbone()
            logger.info("Backbone unfrozen: trainable parameters now %s", f"{count_parameters(model):,}")

        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        for key, value in zip(history, (train_loss, train_acc, val_loss, val_acc)):
            history[key].append(value)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

        logger.info(
            "Epoch [%02d/%d] train loss %.4f acc %.2f%% | val loss %.4f acc %.2f%% | "
            "lr %.2e | %.1fs%s",
            epoch, args.epochs, train_loss, train_acc, val_loss, val_acc,
            optimizer.param_groups[0]["lr"], time.time() - t0,
            "  <-- new best, saved" if improved else "",
        )

    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
    logger.info("Finished in %.1f min. Best validation accuracy: %.2f%%",
                (time.time() - start) / 60, best_val_acc)
    logger.info("Best weights saved to '%s'", best_path)


if __name__ == "__main__":
    main()