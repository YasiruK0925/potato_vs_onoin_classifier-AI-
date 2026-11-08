import argparse
import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from model import CLASS_NAMES, build_model
from train_model import build_transforms  # same preprocessing as validation

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("app")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MODEL_NAME: str = os.getenv("MODEL_NAME", "mobilenet_v2")
IMG_SIZE: int = int(os.getenv("IMG_SIZE", "224"))
NUM_CLASSES: int = 2
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))

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

# --- Stage 1: color / pixel heuristics (0-255 pixel scale) -------------------
COLOR_FILTER_ENABLED: bool = os.getenv("COLOR_FILTER_ENABLED", "1") == "1"
DARK_MEAN_MAX: float = float(os.getenv("DARK_MEAN_MAX", "12"))          # mean brightness below -> "black"
BRIGHT_MEAN_MIN: float = float(os.getenv("BRIGHT_MEAN_MIN", "245"))     # mean brightness above -> "white"
SOLID_STD_MAX: float = float(os.getenv("SOLID_STD_MAX", "6"))           # per-channel std below -> "flat color"
BLUE_FRACTION_MAX: float = float(os.getenv("BLUE_FRACTION_MAX", "0.55"))  # share of blue pixels -> "sky/ocean"

# --- Stage 2: pre-trained ImageNet gatekeeper --------------------------------
GATEKEEPER_ENABLED: bool = os.getenv("GATEKEEPER_ENABLED", "1") == "1"
GATEKEEPER_WEIGHTS_PATH: str = os.getenv("GATEKEEPER_WEIGHTS_PATH", "")  # optional offline state_dict
# Reject only if the top-1 ImageNet class is NOT food/plant-like AND the total probability mass
# on food/plant-like classes is below this value. Lower = more permissive.
GATEKEEPER_MIN_FOOD_MASS: float = float(os.getenv("GATEKEEPER_MIN_FOOD_MASS", "0.15"))
# Extra ImageNet labels to allow, comma separated (case-insensitive), e.g. "basket,carton"
GATEKEEPER_EXTRA_ALLOWED: set[str] = {
    s.strip().lower() for s in os.getenv("GATEKEEPER_EXTRA_ALLOWED", "").split(",") if s.strip()
}

# ImageNet-1k labels (exact torchvision category names, lowercased) considered acceptable:
# vegetables, fruits, fungi, plants, food, soil/earth-like, and typical grocery/kitchen context.
GATEKEEPER_ALLOWED_LABELS: set[str] = {
    # vegetables
    "head cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash", "acorn squash",
    "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "corn",
    "mashed potato", "ear",
    # fruits / seeds / nuts
    "granny smith", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit",
    "custard apple", "pomegranate", "acorn", "buckeye", "hip",
    # plants / earth
    "hay", "daisy", "rapeseed", "earthstar", "agaric", "bolete", "hen-of-the-woods", "coral fungus",
    "stinkhorn", "gyromitra", "plow",
    # prepared food
    "guacamole", "consomme", "hot pot", "dough", "meat loaf", "carbonara", "french loaf", "bagel",
    "pretzel", "pizza", "burrito", "potpie",
    # grocery / kitchen context in which potatoes & onions are commonly photographed
    "grocery store", "shopping basket", "plastic bag", "mixing bowl", "soup bowl", "plate", "pot",
    "frying pan", "wok", "crock pot", "caldron", "wooden spoon", "carton", "crate", "bucket", "hamper",
    "tray", "butcher shop", "wheelbarrow",
}

# --- Stage 3: feature-space anomaly check (needs one-time calibration) --------
FEATURE_CHECK_ENABLED: bool = os.getenv("FEATURE_CHECK_ENABLED", "1") == "1"
FEATURE_STATS_FILE = Path(os.getenv("FEATURE_STATS_FILE", "outputs/feature_stats.json"))
# Reject if distance to the predicted class centroid > MULT * (99th percentile of training distances).
FEATURE_DIST_MULT: float = float(os.getenv("FEATURE_DIST_MULT", "1.5"))


class ModelLoadError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #
app = FastAPI(title="Potato vs Onion API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# --------------------------------------------------------------------------- #
# Stage 1: color / pixel heuristics
# --------------------------------------------------------------------------- #
def check_color_heuristics(image: Image.Image) -> str | None:
    """Return a short description of why the image is unusable, or None if it passes."""
    small = image.resize((64, 64), Image.BILINEAR)
    rgb = np.asarray(small, dtype=np.float32)

    mean_brightness = float(rgb.mean())
    if mean_brightness < DARK_MEAN_MAX:
        return "almost completely black"
    if mean_brightness > BRIGHT_MEAN_MIN:
        return "almost completely white"

    channel_std = float(rgb.std(axis=(0, 1)).mean())
    if channel_std < SOLID_STD_MAX:
        return "a single flat color"

    # PIL HSV is 0-255 for every channel; blue hues ~190-260 degrees -> ~134-184.
    hsv = np.asarray(small.convert("HSV"), dtype=np.float32)
    hue, sat, val = hsv[..., 0], hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    blue_fraction = float(((hue >= 134) & (hue <= 184) & (sat >= 0.25) & (val >= 0.2)).mean())
    if blue_fraction >= BLUE_FRACTION_MAX:
        return f"overwhelmingly blue ({blue_fraction:.0%} of pixels; looks like sky or water)"
    return None


# --------------------------------------------------------------------------- #
# Model wrapper
# --------------------------------------------------------------------------- #
class Classifier:
    def __init__(self, model_name: str, weights_candidates: tuple[Path, ...], img_size: int, threshold: float) -> None:
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = self._load_class_names()

        # Custom model + hook on the input of its last Linear layer (penultimate features).
        self._features: torch.Tensor | None = None
        self.model = self._load_model(model_name, weights_candidates)
        self._register_feature_hook()
        _, self.transform = build_transforms(img_size)

        # Gatekeeper (loaded once, at startup).
        self.gatekeeper: nn.Module | None = None
        if GATEKEEPER_ENABLED:
            self._load_gatekeeper()

        # Optional feature statistics for the anomaly check.
        self.feature_stats = self._load_feature_stats() if FEATURE_CHECK_ENABLED else None

    # ----------------------------- loading ---------------------------------- #
    @staticmethod
    def _load_class_names() -> list[str]:
        names: list[str] | None = None
        source = "model.CLASS_NAMES"
        if CLASSES_FILE.is_file():
            try:
                names = list(json.loads(CLASSES_FILE.read_text()))
                source = str(CLASSES_FILE)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read %s (%s); using defaults.", CLASSES_FILE, exc)
        if names is None:
            names = list(CLASS_NAMES)
        if len(names) != NUM_CLASSES:
            raise ModelLoadError("Expected 2 classes")
        return names

    def _load_model(self, name: str, candidates: tuple[Path, ...]) -> nn.Module:
        weights = next((p for p in candidates if p.is_file()), None)
        if weights is None:
            raise ModelLoadError("No trained weights found.")
        kwargs = {"pretrained": False} if name == "mobilenet_v2" else {}
        model = build_model(name, num_classes=NUM_CLASSES, **kwargs)
        state = torch.load(weights, map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        return model.to(self.device).eval()

    def _register_feature_hook(self) -> None:
        """Capture the input of the final nn.Linear (= penultimate feature vector) on every forward pass."""
        last_linear: nn.Linear | None = None
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                last_linear = module
        if last_linear is None:
            logger.warning("No nn.Linear layer found in custom model; feature anomaly check disabled.")
            return

        def _hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            self._features = inputs[0].detach()

        last_linear.register_forward_pre_hook(_hook)

    def _load_gatekeeper(self) -> None:
        try:
            if GATEKEEPER_WEIGHTS_PATH and Path(GATEKEEPER_WEIGHTS_PATH).is_file():
                gk = models.mobilenet_v2(weights=None)
                gk.load_state_dict(torch.load(GATEKEEPER_WEIGHTS_PATH, map_location="cpu", weights_only=True))
            else:
                # Downloads ~14 MB on first run, then served from the torch hub cache.
                gk = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        except Exception as exc:  # network / file problems
            raise ModelLoadError(
                f"Could not load gatekeeper model ({exc}). Set GATEKEEPER_WEIGHTS_PATH for offline use "
                "or GATEKEEPER_ENABLED=0 to disable."
            ) from exc

        self.gatekeeper = gk.to(self.device).eval()
        self.gk_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.gk_categories: list[str] = list(models.MobileNet_V2_Weights.IMAGENET1K_V1.meta["categories"])

        wanted = GATEKEEPER_ALLOWED_LABELS | GATEKEEPER_EXTRA_ALLOWED
        lowered = [c.lower() for c in self.gk_categories]
        self.gk_allowed_set: set[int] = {i for i, c in enumerate(lowered) if c in wanted}
        missing = sorted(wanted - set(lowered))
        if missing:
            logger.warning("Gatekeeper labels not found in ImageNet categories (ignored): %s", missing)
        self.gk_allowed_idx = torch.tensor(sorted(self.gk_allowed_set), dtype=torch.long)
        logger.info("Gatekeeper ready: %d allowed ImageNet classes.", len(self.gk_allowed_set))

    def _load_feature_stats(self) -> dict[str, Any] | None:
        if not FEATURE_STATS_FILE.is_file():
            logger.warning(
                "%s not found: feature anomaly check is INACTIVE. Run `python app.py --calibrate <train_dir>` once.",
                FEATURE_STATS_FILE,
            )
            return None
        try:
            raw = json.loads(FEATURE_STATS_FILE.read_text())
            return {
                "centroids": np.asarray(raw["centroids"], dtype=np.float32),
                "p99": np.asarray(raw["p99"], dtype=np.float32),
            }
        except (json.JSONDecodeError, KeyError, OSError, ValueError) as exc:
            logger.warning("Could not read %s (%s); feature anomaly check inactive.", FEATURE_STATS_FILE, exc)
            return None

    # ----------------------------- inference -------------------------------- #
    @torch.no_grad()
    def _forward(self, image: Image.Image) -> tuple[str, int, float, dict[str, float], np.ndarray | None]:
        batch = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        self._features = None
        probs = torch.softmax(self.model(batch), dim=1).squeeze(0).cpu()
        confidence, idx = probs.max(dim=0)
        scores = {name: float(p) for name, p in zip(self.class_names, probs)}
        feats = None
        if self._features is not None:
            feats = self._features.flatten(start_dim=1)[0].cpu().numpy()
        return self.class_names[int(idx)], int(idx), float(confidence), scores, feats

    def predict(self, image: Image.Image) -> tuple[str, float, dict[str, float]]:
        """Raw custom-model prediction (no validation pipeline)."""
        name, _idx, confidence, scores, _feats = self._forward(image)
        return name, confidence, scores

    @torch.no_grad()
    def _gatekeeper_check(self, image: Image.Image) -> tuple[bool, str, float]:
        """Returns (passed, top ImageNet label, probability mass on acceptable classes)."""
        assert self.gatekeeper is not None
        batch = self.gk_transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.gatekeeper(batch), dim=1).squeeze(0).cpu()
        top_idx = int(probs.argmax())
        food_mass = float(probs[self.gk_allowed_idx].sum()) if len(self.gk_allowed_idx) else 0.0
        passed = top_idx in self.gk_allowed_set or food_mass >= GATEKEEPER_MIN_FOOD_MASS
        return passed, self.gk_categories[top_idx], food_mass

    def _feature_anomaly(self, feats: np.ndarray | None, class_idx: int) -> tuple[bool, float, float]:
        """Returns (is_anomalous, distance, limit). Inactive (False) when not calibrated."""
        if self.feature_stats is None or feats is None:
            return False, 0.0, 0.0
        centroid = self.feature_stats["centroids"][class_idx]
        limit = float(self.feature_stats["p99"][class_idx]) * FEATURE_DIST_MULT
        dist = float(np.linalg.norm(feats - centroid))
        return (not np.isfinite(dist)) or dist > limit, dist, limit

    # ----------------------------- pipeline --------------------------------- #
    def _reject(self, stage: str, message: str) -> dict[str, Any]:
        logger.info("Rejected at %s: %s", stage, message)
        return {
            "class": "Unknown",
            "prediction": "Unknown",
            "confidence": 0.0,
            "probabilities": {name: 0.0 for name in self.class_names},
            "top_class": "Unknown",
            "is_unknown": True,
            "message": message,
            "rejected_by": stage,
        }

    def analyze(self, image: Image.Image) -> dict[str, Any]:
        """Defense-in-depth validation + classification. Returns the frontend JSON payload."""
        # 1) color / pixel heuristics
        if COLOR_FILTER_ENABLED:
            reason = check_color_heuristics(image)
            if reason:
                return self._reject("color_filter", f"Image rejected by color filter: image is {reason}.")

        # 2) pre-trained ImageNet gatekeeper
        if self.gatekeeper is not None:
            passed, top_label, food_mass = self._gatekeeper_check(image)
            if not passed:
                return self._reject(
                    "gatekeeper",
                    f"Image rejected by Gatekeeper: not a vegetable (looks like '{top_label}').",
                )

        # Custom model forward pass (also yields penultimate features via hook)
        class_name, class_idx, confidence, scores, feats = self._forward(image)
        probabilities = {k: round(v * 100, 2) for k, v in scores.items()}

        # 3) feature-space anomaly check
        if FEATURE_CHECK_ENABLED:
            anomalous, dist, limit = self._feature_anomaly(feats, class_idx)
            if anomalous:
                logger.info("Feature anomaly: distance %.3f > limit %.3f", dist, limit)
                return {
                    "class": "Unknown",
                    "prediction": "Unknown",
                    "confidence": round(confidence * 100, 2),
                    "probabilities": probabilities,
                    "top_class": class_name,
                    "is_unknown": True,
                    "message": "Image rejected by feature anomaly check: image looks unlike any potato or onion "
                               "the model was trained on.",
                    "rejected_by": "feature_anomaly",
                }

        # 4) strict confidence threshold (final verdict)
        is_unknown = confidence < self.threshold
        label = "Unknown" if is_unknown else class_name
        return {
            "class": label,
            "prediction": label,
            "confidence": round(confidence * 100, 2),
            "probabilities": probabilities,
            "top_class": class_name,
            "is_unknown": is_unknown,
            "message": "Please upload a valid image of a potato or an onion." if is_unknown else "OK",
            "rejected_by": "confidence_threshold" if is_unknown else None,
        }

    # ----------------------------- calibration ------------------------------ #
    @torch.no_grad()
    def calibrate_features(self, data_dir: str, out_path: Path = FEATURE_STATS_FILE, max_per_class: int = 1000) -> None:
        """Compute per-class feature centroids + 99th-percentile distances from your EXISTING training
        images (ImageFolder layout: data_dir/<class_name>/*.jpg). No new data is needed."""
        dataset = datasets.ImageFolder(data_dir, transform=self.transform)
        folder_names = [c.lower() for c in dataset.classes]
        model_names = [c.lower() for c in self.class_names]
        if sorted(folder_names) == sorted(model_names):
            to_model_idx = {i: model_names.index(n) for i, n in enumerate(folder_names)}
        else:
            logger.warning("Folder classes %s differ from model classes %s; assuming same order.",
                           dataset.classes, self.class_names)
            to_model_idx = {i: i for i in range(len(dataset.classes))}

        per_class: list[list[np.ndarray]] = [[] for _ in range(NUM_CLASSES)]
        for images, labels in DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0):
            self._features = None
            self.model(images.to(self.device))
            if self._features is None:
                raise RuntimeError("Feature hook did not fire; cannot calibrate.")
            feats = self._features.flatten(start_dim=1).cpu().numpy()
            for f, lab in zip(feats, labels.tolist()):
                bucket = per_class[to_model_idx[lab]]
                if len(bucket) < max_per_class:
                    bucket.append(f)

        centroids, p99s = [], []
        for ci, bucket in enumerate(per_class):
            if not bucket:
                raise RuntimeError(f"No calibration images found for class '{self.class_names[ci]}'.")
            arr = np.stack(bucket)
            centroid = arr.mean(axis=0)
            dists = np.linalg.norm(arr - centroid, axis=1)
            centroids.append(centroid.tolist())
            p99s.append(float(np.percentile(dists, 99)))
            logger.info("%s: n=%d, mean dist=%.3f, p99=%.3f", self.class_names[ci], len(arr), dists.mean(), p99s[-1])

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"class_names": self.class_names, "centroids": centroids, "p99": p99s}))
        logger.info("Saved feature statistics to %s. Restart the server to activate the anomaly check.", out_path)


try:
    classifier = Classifier(MODEL_NAME, WEIGHTS_CANDIDATES, IMG_SIZE, CONFIDENCE_THRESHOLD)
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
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
        return classifier.analyze(image)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", metavar="TRAIN_DIR",
                        help="Build outputs/feature_stats.json from existing training images and exit.")
    args = parser.parse_args()
    if args.calibrate:
        classifier.calibrate_features(args.calibrate)
    else:
        uvicorn.run(app, host="127.0.0.1", port=8000)