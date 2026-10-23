import io
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

from model import CLASS_NAMES, build_model
from train_model import build_transforms  # same preprocessing as validation

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("app")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MODEL_NAME: str = os.getenv("MODEL_NAME", "mobilenet_v2")
IMG_SIZE: int = int(os.getenv("IMG_SIZE", "224"))
# Predictions below this confidence (0-1) are reported as "Unknown".
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))

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
FRONTEND_FILE = Path(__file__).parent / "frontend" / "potato_vs_onion_classifier_frontend.html"


class ModelLoadError(RuntimeError):
    """Raised when the trained model cannot be loaded."""


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #
app = FastAPI(title="Potato vs Onion API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Model wrapper
# --------------------------------------------------------------------------- #
class Classifier:
    """Loads the trained model once and serves predictions."""

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


# Load the model into memory once at startup
try:
    classifier = Classifier(MODEL_NAME, WEIGHTS_CANDIDATES, IMG_SIZE, CONFIDENCE_THRESHOLD)
    logger.info("Model loaded successfully. Ready for API requests!")
except ModelLoadError as exc:
    logger.error("%s", exc)
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/")
async def serve_home():
    return FileResponse(FRONTEND_FILE)


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        class_name, confidence, scores = classifier.predict(image)

        is_unknown = confidence < classifier.threshold
        label = "Unknown" if is_unknown else class_name

        return {
            "class": label,                                   # key the frontend reads
            "prediction": label,                              # kept for compatibility
            "confidence": round(confidence * 100, 2),         # number, 0-100
            "probabilities": {k: round(v * 100, 2) for k, v in scores.items()},  # 0-100
            "top_class": class_name,
            "is_unknown": is_unknown,
            "message": (
                "Please upload a valid image of a potato or an onion."
                if is_unknown
                else "OK"
            ),
        }
    except Exception as exc:
        logger.exception("Error during prediction")
        return JSONResponse(status_code=500, content={"error": str(exc)})


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logger.info("Starting the Backend API Server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)