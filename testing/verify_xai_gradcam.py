"""XAI-002 verification for read-only Grad-CAM artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["KERAS_HOME"] = str(PROJECT_ROOT / ".keras")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config  # noqa: E402
import xai  # noqa: E402


PROTECTED_ARTIFACTS = (
    config.MODEL_PATH, config.CLASS_NAMES_PATH, config.TRAINING_HISTORY_PATH,
    config.EXPERIMENT_CONFIG_PATH, config.TEST_METRICS_PATH,
    config.CLASSIFICATION_REPORT_PATH, config.CONFUSION_MATRIX_CSV_PATH,
    config.TRAINING_ACCURACY_FIG_PATH, config.TRAINING_LOSS_FIG_PATH,
    config.CONFUSION_MATRIX_FIG_PATH,
    PROJECT_ROOT / "experiments" / "context_test_records.json",
    PROJECT_ROOT / "reports" / "context" / "m2_verification.json",
    PROJECT_ROOT / "reports" / "context" / "m2_evidence_audit.json",
)


def hashes() -> dict[str, str]:
    return {
        str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in PROTECTED_ARTIFACTS
    }


def main() -> None:
    before = hashes()
    image_path = PROJECT_ROOT / "test_leaf.jpg"
    metadata = xai.generate_gradcam_evidence(image_path)
    raw_heatmap = np.load(metadata["artifacts"]["raw_heatmap"])
    resized_heatmap = np.load(metadata["artifacts"]["resized_heatmap"])
    with Image.open(metadata["artifacts"]["overlay"]) as overlay:
        overlay.verify()
    with Image.open(metadata["artifacts"]["overlay"]) as overlay:
        overlay_mode, overlay_size = overlay.mode, overlay.size
    after = hashes()
    prediction = metadata["model1_prediction"]
    checks = {
        "model_and_class_mapping_loaded": prediction["output_dimension"] == 38,
        "test_image_exists": image_path.is_file(),
        "prediction_generated": isinstance(prediction["predicted_label"], str),
        "target_matches_prediction": (
            metadata["target_class"] == prediction["predicted_label"]
            and metadata["target_class_index"] == prediction["predicted_index"]
        ),
        "confidence_preserved": 0.0 <= prediction["confidence"] <= 1.0,
        "out_relu_is_spatial_7x7": metadata["target_layer_output_shape"][1:3] == [7, 7],
        "heatmap_finite": bool(np.isfinite(raw_heatmap).all()),
        "heatmap_nonnegative": bool((raw_heatmap >= 0).all()),
        "heatmap_is_7x7": raw_heatmap.shape == (7, 7),
        "resized_heatmap_matches_image": tuple(metadata["resized_heatmap_dimensions"]) == overlay_size,
        "overlay_valid_rgb_or_rgba": overlay_mode in {"RGB", "RGBA"},
        "overlay_matches_image": tuple(metadata["overlay_dimensions"]) == overlay_size,
        "resized_heatmap_finite": bool(np.isfinite(resized_heatmap).all()),
        "protected_artifacts_unchanged": before == after,
    }
    metadata["verification_result"] = checks
    metadata["protected_artifact_hashes_before"] = before
    metadata["protected_artifact_hashes_after"] = after
    xai.write_xai_metadata(metadata)
    record_path = PROJECT_ROOT / "experiments" / "xai_test_records.json"
    record_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
