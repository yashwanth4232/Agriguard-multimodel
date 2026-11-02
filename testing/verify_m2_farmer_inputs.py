"""Verification for the M2-006 farmer-context input data contract."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("KERAS_HOME", str(PROJECT_ROOT / ".keras"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config  # noqa: E402
import context  # noqa: E402
import farmer_inputs  # noqa: E402


PROTECTED_ARTIFACTS = (
    config.MODEL_PATH, config.CLASS_NAMES_PATH, config.TRAINING_HISTORY_PATH,
    config.EXPERIMENT_CONFIG_PATH, config.TEST_METRICS_PATH,
    config.CLASSIFICATION_REPORT_PATH, config.CONFUSION_MATRIX_CSV_PATH,
    config.TRAINING_ACCURACY_FIG_PATH, config.TRAINING_LOSS_FIG_PATH,
    config.CONFUSION_MATRIX_FIG_PATH,
)
EVIDENCE_PATH = PROJECT_ROOT / "reports" / "context" / "farmer_input_verification.json"


def hashes() -> dict[str, str]:
    return {
        str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in PROTECTED_ARTIFACTS
    }


def rejected(callback) -> bool:
    try:
        callback()
    except context.ContextValidationError:
        return True
    return False


def main() -> None:
    before = hashes()
    stages = {
        stage: farmer_inputs.select_growth_stage(stage)
        for stage in farmer_inputs.FARMER_GROWTH_STAGE_DESCRIPTIONS
    }
    confirmed_gps = farmer_inputs.device_gps_location(
        17.3850, 78.4867, at_crop_location_confirmed=True
    )
    user_selected = farmer_inputs.user_selected_location(17.3850, 78.4867)
    unavailable = farmer_inputs.unavailable_crop_location()
    farmer_input = farmer_inputs.build_farmer_context_input("Flowering", None)
    model1_prediction = {
        "predicted_index": 29,
        "predicted_label": "Tomato___Early_blight",
        "confidence": 0.9400433,
        "output_dimension": 38,
    }
    validated_prediction = context.validate_model1_prediction(model1_prediction)
    after = hashes()
    checks = {
        "all_six_growth_stages_accepted": set(stages) == set(context.GROWTH_STAGE_DESCRIPTIONS),
        "growth_stage_descriptions_preserved": all(
            selection["growth_stage_description"]
            == farmer_inputs.FARMER_GROWTH_STAGE_DESCRIPTIONS[stage]
            for stage, selection in stages.items()
        ),
        "invalid_growth_stage_rejected": rejected(
            lambda: farmer_inputs.select_growth_stage("Bud")
        ),
        "confirmed_device_gps_accepted": confirmed_gps["location_source"] == "device_gps",
        "unconfirmed_device_gps_rejected": rejected(
            lambda: farmer_inputs.device_gps_location(17.3850, 78.4867, at_crop_location_confirmed=False)
        ),
        "user_selected_location_accepted": user_selected["location_source"] == "user_selected",
        "user_selected_needs_no_gps_confirmation": user_selected["location_status"] == "available",
        "unavailable_location_has_null_coordinates": (
            unavailable["location_status"] == "unavailable"
            and unavailable["latitude"] is None and unavailable["longitude"] is None
        ),
        "unavailable_location_does_not_lookup_weather": "weather_status" not in farmer_input,
        "invalid_latitude_rejected": rejected(
            lambda: farmer_inputs.user_selected_location(91, 0)
        ),
        "invalid_longitude_rejected": rejected(
            lambda: farmer_inputs.user_selected_location(0, 181)
        ),
        "no_computer_or_uploader_location_input": "computer_location" not in farmer_inputs.__dict__,
        "model1_prediction_unchanged": validated_prediction == model1_prediction,
        "model1_confidence_unchanged": validated_prediction["confidence"] == 0.9400433,
        "class_mapping_unchanged": len(context.load_authoritative_class_names()) == 38,
        "protected_model1_artifacts_unchanged": before == after,
    }
    evidence = {
        "test_type": "M2-006 farmer context input contract verification",
        "controlled_coordinates": {"latitude": 17.3850, "longitude": 78.4867},
        "growth_stage_selections": stages,
        "location_modes": {
            "confirmed_device_gps": confirmed_gps,
            "user_selected": user_selected,
            "unavailable": unavailable,
        },
        "checks": checks,
        "protected_artifact_hashes_before": before,
        "protected_artifact_hashes_after": after,
        "limitations": (
            "Desktop verification validates only the data contract. It does not obtain "
            "phone GPS, access a map picker, or use a computer/uploader location."
        ),
    }
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
