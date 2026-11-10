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
from torchvision import models, transforms

from model import CLASS_NAMES, build_model
from train_model import MultiLabelImageDataset, build_transforms, describe_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("app")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MODEL_NAME: str = os.getenv("MODEL_NAME", "mobilenet_v2")
IMG_SIZE: int = int(os.getenv("IMG_SIZE", "224"))
NUM_CLASSES: int = 2
# Multi-label decision rule: a class is "detected" when its sigmoid probability is above this.
DETECTION_THRESHOLD: float = float(os.getenv("DETECTION_THRESHOLD", "0.5"))

WEIGHTS_CANDIDATES: tuple[Path, ...] = tuple(
    Path(p)
    for p in (
        os.getenv("WEIGHTS_PATH", ""),
        "outputs/best_model.pth",
    )
    if p
)
CLASSES_FILE = Path("outputs/classes.json")
FRONTEND_FILE = Path(__file__).parent / "frontend" / "potato_vs_onion_classifier_frontend.html"

# Friendly singular names used in the "detected" list.
DISPLAY_NAMES = {"onions": "onion", "potatoes": "potato"}


def display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


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
    "tray", "butcher shop",
}

# --- Stage 3: feature-space anomaly check (needs one-time calibration) --------
FEATURE_CHECK_ENABLED: bool = os.getenv("FEATURE_CHECK_ENABLED", "1") == "1"
FEATURE_STATS_FILE = Path(os.getenv("FEATURE_STATS_FILE", "outputs/feature_stats.json"))
# Reject if the distance to the NEAREST known label-combination centroid is greater than
# MULT * (99th percentile of the training distances for that combination).
FEATURE_DIST_MULT: float = float(os.getenv("FEATURE_DIST_MULT", "1.5"))
FEATURE_STATS_VERSION = 2


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
        if CLASSES_FILE.is_file():
            try:
                names = list(json.loads(CLASSES_FILE.read_text()))
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
            raise ModelLoadError(f"No trained weights found. Looked in: {[str(p) for p in candidates]}")
        logger.info("Loading weights from %s", weights.resolve())
        kwargs = {"pretrained": False} if name == "mobilenet_v2" else {}
        model = build_model(name, num_classes=NUM_CLASSES, **kwargs)
        state = torch.load(weights, map_location=self.device, weights_only=True)
        try:
            model.load_state_dict(state)
        except RuntimeError as exc:
            raise ModelLoadError(
                f"{weights} does not match architecture '{name}' (keys look like {list(state)[:4]}). "
                "Train with train_model.py or point WEIGHTS_PATH at the right file."
            ) from exc
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
                "%s not found: feature anomaly check is INACTIVE. Run `python app.py --calibrate <data_dir>` once.",
                FEATURE_STATS_FILE,
            )
            return None
        try:
            raw = json.loads(FEATURE_STATS_FILE.read_text())
            if raw.get("version") != FEATURE_STATS_VERSION:
                logger.warning("%s is from an older single-label model; feature anomaly check INACTIVE. "
                               "Re-run `python app.py --calibrate <data_dir>`.", FEATURE_STATS_FILE)
                return None
            return {
                "centroids": np.asarray(raw["centroids"], dtype=np.float32),  # (groups, feature_dim)
                "p99": np.asarray(raw["p99"], dtype=np.float32),              # (groups,)
            }
        except (json.JSONDecodeError, KeyError, OSError, ValueError) as exc:
            logger.warning("Could not read %s (%s); feature anomaly check inactive.", FEATURE_STATS_FILE, exc)
            return None

    # ----------------------------- inference -------------------------------- #
    @torch.no_grad()
    def _forward(self, image: Image.Image) -> tuple[dict[str, float], np.ndarray | None]:
        """Run the custom model. Returns independent sigmoid probabilities per class (0.0 - 1.0)
        and the penultimate feature vector."""
        batch = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        self._features = None
        probs = torch.sigmoid(self.model(batch)).squeeze(0).cpu()  # sigmoid, NOT softmax
        scores = {name: float(p) for name, p in zip(self.class_names, probs)}
        feats = None
        if self._features is not None:
            feats = self._features.flatten(start_dim=1)[0].cpu().numpy()
        return scores, feats

    def predict(self, image: Image.Image) -> dict[str, float]:
        """Raw custom-model probabilities per class (no validation pipeline)."""
        scores, _feats = self._forward(image)
        return scores

    @torch.no_grad()
    def _gatekeeper_check(self, image: Image.Image) -> tuple[bool, str, float]:
        """Returns (passed, top ImageNet label, probability mass on acceptable classes).
        ImageNet is a single-label model, so softmax is correct here."""
        assert self.gatekeeper is not None
        batch = self.gk_transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.gatekeeper(batch), dim=1).squeeze(0).cpu()
        top_idx = int(probs.argmax())
        food_mass = float(probs[self.gk_allowed_idx].sum()) if len(self.gk_allowed_idx) else 0.0
        passed = top_idx in self.gk_allowed_set or food_mass >= GATEKEEPER_MIN_FOOD_MASS
        return passed, self.gk_categories[top_idx], food_mass

    def _feature_anomaly(self, feats: np.ndarray | None) -> tuple[bool, float, float]:
        """Returns (is_anomalous, distance, limit) measured against the NEAREST known label combination
        (onion only / potato only / both). Inactive (False) when not calibrated."""
        if self.feature_stats is None or feats is None:
            return False, 0.0, 0.0
        if not np.isfinite(feats).all():
            return True, float("inf"), 0.0
        limits = self.feature_stats["p99"] * FEATURE_DIST_MULT
        dists = np.linalg.norm(self.feature_stats["centroids"] - feats[None, :], axis=1)
        ratios = dists / np.maximum(limits, 1e-8)
        best = int(ratios.argmin())
        return bool(ratios[best] > 1.0), float(dists[best]), float(limits[best])

    # ----------------------------- pipeline --------------------------------- #
    def _reject(self, stage: str, message: str) -> dict[str, Any]:
        """Early rejection (model never ran): nothing detected."""
        logger.info("Rejected at %s: %s", stage, message)
        return {
            "class": "Unknown",
            "prediction": "Unknown",
            "confidence": 0.0,
            "probabilities": {name: 0.0 for name in self.class_names},
            "top_class": "Unknown",
            "detected": [],
            "detections": [],
            "is_unknown": True,
            "message": message,
            "rejected_by": stage,
        }

    def _response(self, scores: dict[str, float], detected: list[str], message: str,
                  rejected_by: str | None) -> dict[str, Any]:
        """Build the JSON payload from real model scores.

        ``detected`` holds the class names to report (empty -> nothing detected / unknown).
        Legacy fields (class, prediction, confidence, probabilities, top_class, is_unknown, message)
        are kept so the existing frontend does not break; ``detected`` and ``detections`` are new.
        """
        top_name = max(scores, key=scores.get)
        detections = [
            {"name": display_name(n), "confidence": round(scores[n] * 100, 2)} for n in detected
        ]
        label = " + ".join(detected) if detected else "Unknown"
        return {
            "class": label,
            "prediction": label,
            "confidence": round(scores[top_name] * 100, 2),
            "probabilities": {k: round(v * 100, 2) for k, v in scores.items()},
            "top_class": top_name,
            "detected": [d["name"] for d in detections],   # e.g. ["potato", "onion"]
            "detections": detections,                       # e.g. [{"name": "potato", "confidence": 97.31}, ...]
            "is_unknown": not detected,
            "message": message,
            "rejected_by": rejected_by,
        }

    def analyze(self, image: Image.Image) -> dict[str, Any]:
        """Defense-in-depth validation + multi-label classification. Returns the frontend JSON payload."""
        # 1) color / pixel heuristics
        if COLOR_FILTER_ENABLED:
            reason = check_color_heuristics(image)
            if reason:
                return self._reject("color_filter", f"Image rejected by color filter: image is {reason}.")

        # 2) pre-trained ImageNet gatekeeper
        if self.gatekeeper is not None:
            passed, top_label, _food_mass = self._gatekeeper_check(image)
            if not passed:
                return self._reject(
                    "gatekeeper",
                    f"Image rejected by Gatekeeper: not a vegetable (looks like '{top_label}').",
                )

        # Custom model forward pass (also yields penultimate features via hook)
        scores, feats = self._forward(image)

        # 3) feature-space anomaly check
        if FEATURE_CHECK_ENABLED:
            anomalous, dist, limit = self._feature_anomaly(feats)
            if anomalous:
                logger.info("Feature anomaly: distance %.3f > limit %.3f", dist, limit)
                return self._response(
                    scores, [],
                    "Image rejected by feature anomaly check: image looks unlike any potato or onion "
                    "the model was trained on.",
                    "feature_anomaly",
                )

        # 4) multi-label verdict: every class is judged independently (no argmax)
        detected = [
            name for name, p in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            if p > self.threshold
        ]
        if not detected:
            return self._response(
                scores, [], "Please upload a valid image of a potato or an onion.", "confidence_threshold"
            )
        return self._response(scores, detected, "OK", None)

    # ----------------------------- calibration ------------------------------ #
    @torch.no_grad()
    def calibrate_features(self, data_dir: str, out_path: Path = FEATURE_STATS_FILE,
                           min_group_size: int = 5, max_per_group: int = 1000) -> None:
        """Compute feature centroids + 99th-percentile distances from your EXISTING training data
        (data_dir/labels.csv + data_dir/all_images/). Images are grouped by label combination
        (onion only, potato only, both); images with no label are ignored. No new data is needed."""
        data_dir = Path(data_dir)
        dataset = MultiLabelImageDataset(
            data_dir / "labels.csv", data_dir / "all_images", self.class_names, transform=self.transform
        )

        groups: dict[tuple[int, ...], list[np.ndarray]] = {}
        for images, labels in DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0):
            self._features = None
            self.model(images.to(self.device))
            if self._features is None:
                raise RuntimeError("Feature hook did not fire; cannot calibrate.")
            feats = self._features.flatten(start_dim=1).cpu().numpy()
            for f, lab in zip(feats, labels.tolist()):
                key = tuple(int(v > 0.5) for v in lab)
                if not any(key):
                    continue  # "neither" images are negatives, not part of the normal distribution
                bucket = groups.setdefault(key, [])
                if len(bucket) < max_per_group:
                    bucket.append(f)

        names, centroids, p99s = [], [], []
        for key, bucket in sorted(groups.items()):
            group_name = describe_labels(tuple(float(v) for v in key), self.class_names)
            if len(bucket) < min_group_size:
                logger.warning("Skipping '%s': only %d image(s) (need >= %d).", group_name, len(bucket), min_group_size)
                continue
            arr = np.stack(bucket)
            centroid = arr.mean(axis=0)
            dists = np.linalg.norm(arr - centroid, axis=1)
            names.append(group_name)
            centroids.append(centroid.tolist())
            p99s.append(float(np.percentile(dists, 99)))
            logger.info("%s: n=%d, mean dist=%.3f, p99=%.3f", group_name, len(arr), dists.mean(), p99s[-1])

        if not centroids:
            raise RuntimeError("No label group had enough images to calibrate.")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "version": FEATURE_STATS_VERSION,
            "class_names": self.class_names,
            "groups": names,
            "centroids": centroids,
            "p99": p99s,
        }))
        logger.info("Saved feature statistics to %s. Restart the server to activate the anomaly check.", out_path)


try:
    classifier = Classifier(MODEL_NAME, WEIGHTS_CANDIDATES, IMG_SIZE, DETECTION_THRESHOLD)
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
    parser.add_argument("--calibrate", metavar="DATA_DIR",
                        help="Build outputs/feature_stats.json from DATA_DIR/labels.csv + DATA_DIR/all_images and exit.")
    args = parser.parse_args()
    if args.calibrate:
        classifier.calibrate_features(args.calibrate)
    else:
        uvicorn.run(app, host="127.0.0.1", port=8000)