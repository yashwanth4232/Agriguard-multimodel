"""Verification for M1-008 local single-image inference."""

from __future__ import annotations

import hashlib
import json
import os
import py_compile
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["KERAS_HOME"] = str(PROJECT_ROOT / ".keras")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config  # noqa: E402
import dataset  # noqa: E402
import predict  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    model_hash_before = sha256_file(config.MODEL_PATH)
    class_names_hash_before = sha256_file(config.CLASS_NAMES_PATH)
    py_compile.compile(str(PROJECT_ROOT / "src" / "predict.py"), doraise=True)

    trained_model, class_names = predict.load_prediction_artifacts()
    sample_record = dataset.get_split_records("validation")[0]
    sample_path = PROJECT_ROOT / sample_record["relative_path"]
    result = predict.predict_image(sample_path, trained_model, class_names)

    missing_path = PROJECT_ROOT / "tmp" / "m1_008_missing_image.jpg"
    try:
        predict.predict_image(missing_path, trained_model, class_names)
    except FileNotFoundError as exc:
        missing_image_failure_pass = "Image file not found" in str(exc)
    else:
        missing_image_failure_pass = False

    model_hash_after = sha256_file(config.MODEL_PATH)
    class_names_hash_after = sha256_file(config.CLASS_NAMES_PATH)
    checks = {
        "syntax_check_pass": True,
        "saved_model_loaded_pass": trained_model.output_shape == (None, 38),
        "class_names_loaded_pass": len(class_names) == 38,
        "inference_output_dimension_pass": result["output_dimension"] == 38,
        "predicted_label_pass": result["predicted_label"] in class_names,
        "confidence_probability_pass": 0.0 <= result["confidence"] <= 1.0,
        "nonexistent_image_failure_pass": missing_image_failure_pass,
        "model_artifact_unchanged_pass": model_hash_before == model_hash_after,
        "class_names_artifact_unchanged_pass": (
            class_names_hash_before == class_names_hash_after
        ),
    }
    print(json.dumps({"sample_prediction": result, "checks": checks}, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
