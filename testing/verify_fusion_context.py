"""Controlled, no-network verification of deterministic fusion."""

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
import fusion  # noqa: E402

PROTECTED = (
    config.MODEL_PATH, config.CLASS_NAMES_PATH, config.TRAINING_HISTORY_PATH,
    config.EXPERIMENT_CONFIG_PATH, config.TEST_METRICS_PATH, config.CLASSIFICATION_REPORT_PATH,
    config.CONFUSION_MATRIX_CSV_PATH, config.TRAINING_ACCURACY_FIG_PATH,
    config.TRAINING_LOSS_FIG_PATH, config.CONFUSION_MATRIX_FIG_PATH,
    PROJECT_ROOT / "experiments" / "context_test_records.json",
    PROJECT_ROOT / "reports" / "context" / "m2_verification.json",
    PROJECT_ROOT / "reports" / "context" / "m2_evidence_audit.json",
    PROJECT_ROOT / "experiments" / "xai_test_records.json",
    PROJECT_ROOT / "reports" / "xai" / "gradcam_raw_heatmap.npy",
    PROJECT_ROOT / "reports" / "xai" / "gradcam_resized_heatmap.npy",
    PROJECT_ROOT / "reports" / "xai" / "gradcam_overlay.png",
    PROJECT_ROOT / "reports" / "xai" / "xai_verification.json",
)

def hashes():
    return {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in PROTECTED}

def rejects(callback):
    try: callback()
    except fusion.FusionValidationError: return True
    return False

def main():
    before = hashes()
    end_to_end = json.loads((PROJECT_ROOT / "reports/context/end_to_end_m1_m2_test.json").read_text(encoding="utf-8"))
    xai = json.loads((PROJECT_ROOT / "reports/xai/xai_verification.json").read_text(encoding="utf-8"))
    prediction, record = end_to_end["model1_prediction"], end_to_end["normalized_agricultural_context"]
    output = fusion.fuse_evidence(prediction, record, xai, model_artifact_sha256=before[str(config.MODEL_PATH.relative_to(PROJECT_ROOT))])
    unavailable = dict(record); unavailable.update({"growth_stage":"Unknown", "location_status":"unavailable", "location_source":"unavailable", "latitude":None, "longitude":None, "weather_status":"unavailable", "temperature_c":None, "humidity_percent":None, "precipitation_mm":None, "weather_condition":None, "context_timestamp":None, "weather_raw_response":None})
    no_xai = fusion.fuse_evidence(prediction, unavailable, None)
    wrong_crop = dict(record); wrong_crop["crop"] = "Potato"
    wrong_disease = dict(record); wrong_disease["disease"] = "Tomato___Late_blight"
    wrong_xai_class = dict(xai); wrong_xai_class["target_class"] = "Tomato___Late_blight"
    wrong_xai_index = dict(xai); wrong_xai_index["target_class_index"] = 12
    after = hashes()
    checks = {
        "valid_model1_and_authoritative_mapping": output["diagnosis"]["predicted_label"] == prediction["predicted_label"],
        "crop_and_disease_consistent": output["diagnosis"]["crop"] == record["crop"] and output["diagnosis"]["disease"] == record["disease"],
        "confidence_exactly_preserved": output["diagnosis"]["confidence"] == prediction["confidence"],
        "index_and_label_preserved": output["diagnosis"]["predicted_index"] == prediction["predicted_index"],
        "valid_context_growth_location_weather_preserved": output["agricultural_context"] == record,
        "unknown_and_unavailable_context_accepted": no_xai["availability"] == {"weather_available":False,"location_available":False,"growth_stage_known":False,"xai_available":False},
        "partial_context_nulls_preserved": no_xai["agricultural_context"]["temperature_c"] is None,
        "xai_available_and_consistent": output["visual_explanation"]["available"] is True,
        "xai_unavailable_accepted": no_xai["visual_explanation"]["available"] is False,
        "inconsistent_crop_rejected": rejects(lambda: fusion.fuse_evidence(prediction, wrong_crop)),
        "inconsistent_disease_rejected": rejects(lambda: fusion.fuse_evidence(prediction, wrong_disease)),
        "inconsistent_xai_class_rejected": rejects(lambda: fusion.fuse_evidence(prediction, record, wrong_xai_class)),
        "inconsistent_xai_index_rejected": rejects(lambda: fusion.fuse_evidence(prediction, record, wrong_xai_index)),
        "no_new_diagnosis_or_confidence": output["diagnosis"]["unchanged_by_fusion"] is True,
        "required_advisory_inputs_present": all(k in output["advisory_inputs"] for k in ("disease_label","crop","confidence","growth_stage","current_weather_context_available")),
        "provenance_preserved": output["provenance"]["weather_provider"] == "Open-Meteo",
        "protected_artifacts_unchanged": before == after,
    }
    report_dir = PROJECT_ROOT / "reports/fusion"; report_dir.mkdir(parents=True, exist_ok=True)
    fusion_input = fusion.build_fusion_input(prediction, record, xai, model_artifact_sha256=before[str(config.MODEL_PATH.relative_to(PROJECT_ROOT))])
    for name, value in (("fusion_input.json",fusion_input),("fusion_output.json",output)):
        (report_dir/name).write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")
    evidence={"test_type":"FUSION-002 controlled deterministic evidence association", "checks":checks, "protected_hashes_before":before, "protected_hashes_after":after, "no_api_call":True, "no_training":True, "no_new_confidence_or_diagnosis":True}
    (report_dir/"fusion_verification.json").write_text(json.dumps(evidence,indent=2)+"\n",encoding="utf-8")
    (PROJECT_ROOT/"experiments/fusion_test_records.json").write_text(json.dumps(evidence,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(evidence,indent=2))
    if not all(checks.values()): raise SystemExit(1)

if __name__ == "__main__": main()
