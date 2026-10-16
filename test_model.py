"""Classify a single image with the trained Potato vs Onion model.

Example:
    python test_model.py --image path/to/image.jpg
    python test_model.py --image photo.png --model cnn --img-size 128 \
        --weights outputs/best_model.pth

The ``--model`` and ``--img-size`` values must match what was used in
``train_model.py``, otherwise the weights will not load or accuracy will drop.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError

from model import CLASS_NAMES, build_model
from train_model import build_transforms  # reuse the exact validation preprocessing

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Classify an image as potatoes/onions/both/others.")
    p.add_argument("--image", type=Path, required=True, help="Path to the image to classify.")
    p.add_argument("--weights", type=Path, default=Path("outputs/best_model.pth"),
                   help="Path to the trained weights (.pth).")
    p.add_argument("--classes-file", type=Path, default=Path("outputs/classes.json"),
                   help="JSON file with class names saved during training.")
    p.add_argument("--model", choices=["cnn", "mobilenet_v2"], default="mobilenet_v2")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--top-k", type=int, default=4,
                   help="Number of top predictions to show (1 = top prediction only).")
    return p.parse_args()


def fail(message: str) -> None:
    """Print an error message to stderr and exit with status 1."""
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


def load_class_names(path: Path) -> list[str]:
    """Load class names saved by training, falling back to ``CLASS_NAMES``."""
    if path.is_file():
        try:
            return list(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError) as exc:
            fail(f"Could not read class names from '{path}': {exc}")
    print(f"Note: '{path}' not found, using default class order {CLASS_NAMES}.")
    return list(CLASS_NAMES)


def load_model(name: str, weights: Path, num_classes: int, device: torch.device) -> nn.Module:
    """Build the architecture, load trained weights and switch to eval mode."""
    if not weights.is_file():
        fail(f"Weights file not found: '{weights}'. Train the model first with train_model.py.")

    kwargs = {"pretrained": False} if name == "mobilenet_v2" else {}
    model = build_model(name, num_classes=num_classes, **kwargs)
    try:
        state_dict = torch.load(weights, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        fail(f"Weights do not match --model '{name}' with {num_classes} classes.\n{exc}")
    except Exception as exc:  # corrupted or non-checkpoint file
        fail(f"Could not load weights from '{weights}': {exc}")

    return model.to(device).eval()


def load_image(path: Path) -> Image.Image:
    """Open an image as RGB, with friendly errors for common problems."""
    if not path.is_file():
        fail(f"Image file not found: '{path}'")
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        print(f"Warning: '{path.suffix}' is not a common image extension; trying anyway.")
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except UnidentifiedImageError:
        fail(f"'{path}' is not a valid or supported image file.")
    except OSError as exc:
        fail(f"Could not read image '{path}': {exc}")
    raise AssertionError("unreachable")


@torch.no_grad()
def predict(
    model: nn.Module,
    image: Image.Image,
    img_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return class probabilities (shape ``(num_classes,)``) for one image."""
    _, val_transform = build_transforms(img_size)
    batch = val_transform(image).unsqueeze(0).to(device)  # add batch dimension
    logits = model(batch)
    return torch.softmax(logits, dim=1).squeeze(0).cpu()


def print_report(image_path: Path, classes: list[str], probs: torch.Tensor, top_k: int) -> None:
    """Print a formatted prediction report."""
    k = max(1, min(top_k, len(classes)))
    top_probs, top_idx = probs.topk(k)
    width = max(len(c) for c in classes)

    print()
    print("=" * 52)
    print(f" Image      : {image_path}")
    print(f" Prediction : {classes[top_idx[0]].upper()}  ({top_probs[0] * 100:.2f}% confidence)")
    print("-" * 52)
    label = "All class probabilities" if k == len(classes) else f"Top-{k} probabilities"
    print(f" {label}:")
    for p, i in zip(top_probs, top_idx):
        bar = "#" * int(round(p.item() * 30))
        print(f"   {classes[i]:<{width}}  {p * 100:6.2f}%  {bar}")
    print("=" * 52)


def main() -> None:
    """Entry point."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = load_class_names(args.classes_file)
    model = load_model(args.model, args.weights, len(classes), device)
    image = load_image(args.image)
    probs = predict(model, image, args.img_size, device)
    print_report(args.image, classes, probs, args.top_k)


if __name__ == "__main__":
    main()