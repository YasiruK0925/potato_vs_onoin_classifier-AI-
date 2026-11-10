"""Classify a single image with the trained Potato vs Onion multi-label model.

The model answers two independent questions for every image: "is there a potato?"
and "is there an onion?". Each class gets its own sigmoid probability, so an image
can contain a potato, an onion, both, or neither.

Example:
    python test_model.py --image path/to/image.jpg
    python test_model.py --image photo.png --model cnn --img-size 128 \
        --weights outputs/best_model.pth --threshold 0.5

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

# Friendly singular names for the "detected items" list.
DISPLAY_NAMES = {"onions": "onion", "potatoes": "potato"}


def display_name(name: str) -> str:
    """Singular display name for a class (falls back to the class name itself)."""
    return DISPLAY_NAMES.get(name, name)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Detect potatoes and/or onions in an image.")
    p.add_argument("--image", type=Path, required=True, help="Path to the image to classify.")
    p.add_argument("--weights", type=Path, default=Path("outputs/best_model.pth"),
                   help="Path to the trained weights (.pth).")
    p.add_argument("--classes-file", type=Path, default=Path("outputs/classes.json"),
                   help="JSON file with class names saved during training.")
    p.add_argument("--model", choices=["cnn", "mobilenet_v2"], default="mobilenet_v2")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="A class is detected when its sigmoid probability is above this value.")
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
    """Return independent per-class probabilities (shape ``(num_classes,)``) for one image.

    Uses sigmoid, so each value is between 0 and 1 on its own; they do NOT sum to 1.
    """
    _, val_transform = build_transforms(img_size)
    batch = val_transform(image).unsqueeze(0).to(device)  # add batch dimension
    logits = model(batch)
    return torch.sigmoid(logits).squeeze(0).cpu()


def print_report(image_path: Path, classes: list[str], probs: torch.Tensor, threshold: float) -> None:
    """Print a formatted multi-label prediction report."""
    scored = sorted(
        ((name, float(p)) for name, p in zip(classes, probs)),
        key=lambda item: item[1],
        reverse=True,
    )
    detected = [(name, p) for name, p in scored if p > threshold]  # no argmax: every class judged on its own
    width = max(len(c) for c in classes)

    print()
    print("=" * 56)
    print(f" Image    : {image_path}")
    if detected:
        summary = ", ".join(f"{display_name(n)} ({p * 100:.2f}%)" for n, p in detected)
        print(f" Detected : {summary}")
        print(f" Items    : {[display_name(n) for n, _ in detected]}")
    else:
        print(" Detected : nothing (no class above the threshold)")
        print(" Items    : []")
    print("-" * 56)
    print(f" Per-class probabilities (detected when > {threshold * 100:.0f}%):")
    for name, p in scored:
        bar = "#" * int(round(p * 30))
        flag = "  <- detected" if p > threshold else ""
        print(f"   {name:<{width}}  {p * 100:6.2f}%  {bar}{flag}")
    print("=" * 56)


def main() -> None:
    """Entry point."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = load_class_names(args.classes_file)
    model = load_model(args.model, args.weights, len(classes), device)
    image = load_image(args.image)
    probs = predict(model, image, args.img_size, device)
    print_report(args.image, classes, probs, args.threshold)


if __name__ == "__main__":
    main()