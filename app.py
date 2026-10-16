"""Gradio web app for the Potato vs Onion image classifier.

Run locally:
    python app.py
    python app.py --share          # create a temporary public link

Configuration (environment variables, all optional):
    MODEL_NAME          "mobilenet_v2" (default) or "cnn"
    IMG_SIZE            Input resolution used in training (default 224)
    CONFIDENCE_THRESHOLD  Minimum confidence to accept a prediction (default 0.85)
    WEIGHTS_PATH        Explicit path to the trained .pth file
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image

from model import CLASS_NAMES, build_model
from train_model import build_transforms  # same preprocessing as validation

logger = logging.getLogger("app")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MODEL_NAME: str = os.getenv("MODEL_NAME", "mobilenet_v2")
IMG_SIZE: int = int(os.getenv("IMG_SIZE", "224"))
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.85"))

# The first existing path wins.
WEIGHTS_CANDIDATES: tuple[Path, ...] = tuple(
    Path(p)
    for p in (
        os.getenv("WEIGHTS_PATH", ""),
        "outputs/best_model.pth",    
        "potato_onion_model.pth",
    )
    if p
)
CLASSES_FILE = Path("outputs/classes.json")
EXAMPLES_DIR = Path("examples")
OTHERS_LABEL = "others"

DISPLAY_NAMES: dict[str, str] = {
    "potatoes": "Potatoes",
    "onions": "Onions",
    "both": "Potatoes and onions",
    "others": "Neither",
}

CSS = """
.gradio-container {max-width: 960px !important; margin: auto;}
#title {text-align: center;}
#status {font-size: 1.15rem; padding: 12px 16px; border-radius: 10px;}
footer {visibility: hidden;}
"""


class ModelLoadError(RuntimeError):
    """Raised when the trained model cannot be loaded."""


# --------------------------------------------------------------------------- #
# Model wrapper
# --------------------------------------------------------------------------- #
class Classifier:
    """Loads the trained model once and serves predictions.

    Args:
        model_name: Architecture name understood by ``model.build_model``.
        weights_candidates: Paths to try, in order, for the trained weights.
        img_size: Input resolution (must match training).
        threshold: Minimum confidence required to accept a prediction.

    Raises:
        ModelLoadError: If no weights file exists or the weights are invalid.
    """

    def __init__(
        self,
        model_name: str,
        weights_candidates: tuple[Path, ...],
        img_size: int,
        threshold: float,
    ) -> None:
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = self._load_class_names()
        self.model = self._load_model(model_name, weights_candidates)
        _, self.transform = build_transforms(img_size)

    @staticmethod
    def _load_class_names() -> list[str]:
        """Read class names saved by training, falling back to defaults."""
        if CLASSES_FILE.is_file():
            try:
                return list(json.loads(CLASSES_FILE.read_text()))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read %s (%s); using defaults.", CLASSES_FILE, exc)
        return list(CLASS_NAMES)

    def _load_model(self, name: str, candidates: tuple[Path, ...]) -> nn.Module:
        """Build the architecture and load the first available weights file."""
        weights = next((p for p in candidates if p.is_file()), None)
        if weights is None:
            searched = ", ".join(f"'{p}'" for p in candidates)
            raise ModelLoadError(
                f"No trained weights found (looked for: {searched}). "
                "Train a model with train_model.py or set WEIGHTS_PATH."
            )

        kwargs = {"pretrained": False} if name == "mobilenet_v2" else {}
        model = build_model(name, num_classes=len(self.class_names), **kwargs)
        try:
            state = torch.load(weights, map_location=self.device, weights_only=True)
            model.load_state_dict(state)
        except RuntimeError as exc:
            raise ModelLoadError(
                f"'{weights}' does not match MODEL_NAME='{name}' with "
                f"{len(self.class_names)} classes. Weights from an older "
                f"3-class model must be retrained.\n{exc}"
            ) from exc
        except Exception as exc:
            raise ModelLoadError(f"Could not load weights from '{weights}': {exc}") from exc

        logger.info("Loaded %s weights from '%s' on %s", name, weights, self.device)
        return model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, image: Image.Image) -> tuple[str, float, dict[str, float]]:
        """Classify one image.

        Returns:
            ``(top_class, confidence, probabilities)`` where ``probabilities``
            maps every class name to its probability in [0, 1].
        """
        batch = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(batch), dim=1).squeeze(0).cpu()
        confidence, idx = probs.max(dim=0)
        scores = {name: float(p) for name, p in zip(self.class_names, probs)}
        return self.class_names[int(idx)], float(confidence), scores


# --------------------------------------------------------------------------- #
# UI logic
# --------------------------------------------------------------------------- #
def format_result(top_class: str, confidence: float, threshold: float) -> str:
    """Turn a raw prediction into a user-friendly status message."""
    pct = confidence * 100
    if top_class == OTHERS_LABEL:
        return (
            "### 🚫 No potatoes or onions detected\n"
            "This looks like something else. Please upload a clear photo "
            "of potatoes and/or onions."
        )
    if confidence < threshold:
        return (
            "### ⚠️ Not sure about this one\n"
            f"The best guess was **{DISPLAY_NAMES.get(top_class, top_class)}** at "
            f"only {pct:.1f}% confidence (required: {threshold * 100:.0f}%). "
            "Try a brighter, closer photo with the vegetables clearly in view."
        )
    return (
        f"### ✅ {DISPLAY_NAMES.get(top_class, top_class)}\n"
        f"Confidence: **{pct:.1f}%**"
    )


def build_interface(classifier: Classifier) -> gr.Blocks:
    """Create the Gradio Blocks interface around a loaded classifier."""

    def run(image: Image.Image | None) -> tuple[str, dict[str, float]]:
        """Gradio callback: classify an uploaded image."""
        if image is None:
            return "### 📷 Please upload an image first.", {}
        try:
            top_class, confidence, scores = classifier.predict(image)
        except Exception:  # never crash the UI on a bad image
            logger.exception("Prediction failed")
            return "### ❌ Something went wrong processing that image. Please try another.", {}
        return format_result(top_class, confidence, classifier.threshold), scores

    with gr.Blocks(
        title="Potato vs Onion Classifier",
        theme=gr.themes.Soft(primary_hue="amber"),
        css=CSS,
    ) as demo:
        gr.Markdown(
            "# 🥔🧅 Potato vs Onion Classifier\n"
            "Upload a photo and the model will tell you whether it contains "
            "**potatoes**, **onions**, **both**, or **neither**.",
            elem_id="title",
        )

        with gr.Row(equal_height=True):
            with gr.Column():
                image_in = gr.Image(type="pil", label="Your photo", height=360)
                with gr.Row():
                    clear_btn = gr.ClearButton(value="Clear")
                    submit_btn = gr.Button("Classify", variant="primary")
            with gr.Column():
                status = gr.Markdown("### Waiting for an image…", elem_id="status")
                probs_out = gr.Label(
                    label="Class probabilities",
                    num_top_classes=len(classifier.class_names),
                )

        example_files = (
            sorted(str(p) for p in EXAMPLES_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            if EXAMPLES_DIR.is_dir()
            else []
        )
        if example_files:
            gr.Examples(
                examples=[[f] for f in example_files],
                inputs=image_in,
                outputs=[status, probs_out],
                fn=run,
                cache_examples=False,
                label="Try an example",
            )

        with gr.Accordion("Tips for best results", open=False):
            gr.Markdown(
                "- Use a well-lit, in-focus photo with the vegetables filling most of the frame.\n"
                "- Photos of other subjects (people, landscapes, other food) are reported as "
                "*neither*.\n"
                f"- Predictions below **{classifier.threshold * 100:.0f}%** confidence are "
                "flagged as uncertain instead of guessed."
            )

        submit_btn.click(run, inputs=image_in, outputs=[status, probs_out])
        image_in.upload(run, inputs=image_in, outputs=[status, probs_out])
        clear_btn.add([image_in, status, probs_out])

    return demo


def main() -> None:
    """Load the model and launch the web app."""
    parser = argparse.ArgumentParser(description="Launch the Potato/Onion classifier web app.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    try:
        classifier = Classifier(MODEL_NAME, WEIGHTS_CANDIDATES, IMG_SIZE, CONFIDENCE_THRESHOLD)
    except ModelLoadError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    build_interface(classifier).launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()