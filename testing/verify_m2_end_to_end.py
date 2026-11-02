"""One isolated, live Model 1 -> M2 evidence-generation test."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["KERAS_HOME"] = str(PROJECT_ROOT / ".keras")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config  # noqa: E402
import context  # noqa: E402
import predict  # noqa: E402


PROTECTED_ARTIFACTS = (
    config.MODEL_PATH,
    config.CLASS_NAMES_PATH,
    config.TRAINING_HISTORY_PATH,
    config.EXPERIMENT_CONFIG_PATH,
    config.TEST_METRICS_PATH,
    config.CLASSIFICATION_REPORT_PATH,
    config.CONFUSION_MATRIX_CSV_PATH,
    config.TRAINING_ACCURACY_FIG_PATH,
    config.TRAINING_LOSS_FIG_PATH,
    config.CONFUSION_MATRIX_FIG_PATH,
)
EVIDENCE_PATH = PROJECT_ROOT / "reports" / "context" / "end_to_end_m1_m2_test.json"
IMAGE_PATH = PROJECT_ROOT / "test_leaf.jpg"
LOCATION = {"source": "user_selected", "latitude": 17.3850, "longitude": 78.4867}
GROWTH_STAGE = "Flowering"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in PROTECTED_ARTIFACTS
    }


def main() -> None:
    hashes_before = protected_hashes()
    model, class_names = predict.load_prediction_artifacts()
    prediction = predict.predict_image(IMAGE_PATH, model, class_names)
    normalized_context = context.build_context(prediction, GROWTH_STAGE, LOCATION)
    hashes_after = protected_hashes()
    hashes_unchanged = hashes_before == hashes_after

    checks = {
        "image_exists": IMAGE_PATH.is_file(),
        "model1_output_dimension": prediction["output_dimension"] == 38,
        "model1_label_is_authoritative": prediction["predicted_label"] in class_names,
        "model1_confidence_is_probability": 0.0 <= prediction["confidence"] <= 1.0,
        "crop_extracted": normalized_context["crop"] == "Tomato",
        "disease_preserved": normalized_context["disease"] == prediction["predicted_label"],
        "growth_stage_preserved": normalized_context["growth_stage"] == GROWTH_STAGE,
        "location_source_preserved": normalized_context["location_source"] == "user_selected",
        "requested_coordinates_preserved": (
            normalized_context["latitude"] == LOCATION["latitude"]
            and normalized_context["longitude"] == LOCATION["longitude"]
        ),
        "live_weather_available": normalized_context["weather_status"] == "available",
        "provider_recorded": normalized_context["weather_provider"] == "Open-Meteo",
        "raw_weather_response_preserved": normalized_context["weather_raw_response"] is not None,
        "protected_model1_artifacts_unchanged": hashes_unchanged,
    }
    evidence = {
        "test_type": "M2-005 isolated Model 1 to Model 2 end-to-end test",
        "input": {
            "image_path": str(IMAGE_PATH.relative_to(PROJECT_ROOT)),
            "growth_stage": GROWTH_STAGE,
            "location": LOCATION,
        },
        "model1_prediction": prediction,
        "normalized_agricultural_context": normalized_context,
        "checks": checks,
        "protected_artifact_hashes_before": hashes_before,
        "protected_artifact_hashes_after": hashes_after,
    }
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
