"""Multipart integration checks for the three farmer-facing location modes."""
from __future__ import annotations

import hashlib
import io
import os
import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("KERAS_HOME", str(PROJECT_ROOT / ".keras"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import app  # noqa: E402
import farmer_inputs  # noqa: E402

PROTECTED = [
    "models/agriguard_mobilenetv2.keras", "models/class_names.json", "src/predict.py",
    "src/context.py", "src/farmer_inputs.py", "src/xai.py", "src/fusion.py",
    "src/advisory.py", "data/advisory/knowledge_base.json", "data/advisory/source_registry.json",
]


def hashes() -> dict[str, str]:
    return {path: hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest() for path in PROTECTED}


def image_field() -> tuple[io.BytesIO, str]:
    return io.BytesIO((PROJECT_ROOT / "test_leaf.jpg").read_bytes()), "test_leaf.jpg"


def submission(mode: str, **fields: str):
    data = {"growth_stage": "Flowering", "location_mode": mode, **fields, "leaf_image": image_field()}
    return data


def main() -> None:
    before = hashes()
    client = app.app.test_client()
    home = client.get("/")
    home_html = home.get_data(as_text=True)
    labels = app.context.load_authoritative_class_names()
    index = labels.index("Tomato___Early_blight")
    prediction = {"predicted_index": index, "predicted_label": labels[index], "confidence": 0.9400433301925659, "output_dimension": len(labels)}
    validated = app.context.validate_model1_prediction(prediction)
    unavailable = {"location_status": "unavailable", "location_source": "unavailable", "latitude": None, "longitude": None}
    unavailable_context_input = {"source": "unavailable", "latitude": None, "longitude": None}
    fake_xai = {"model_sha256": "test", "artifacts": {"overlay": str(PROJECT_ROOT / "static/generated/test/overlay.png")}}
    with patch.object(app.predict, "load_prediction_artifacts", return_value=(object(), labels)), \
         patch.object(app.predict, "predict_image", return_value=prediction), \
         patch.object(app.xai, "generate_gradcam_evidence", return_value=fake_xai), \
         patch.object(app.context, "build_context", wraps=app.context.build_context) as build_context, \
         patch.object(app.context, "get_weather", return_value=app.context._unavailable_weather("test weather seam")) as weather_request, \
         patch.object(app.fusion, "fuse_evidence", return_value={"fusion_type": "deterministic_context_association"}), \
         patch.object(app.advisory, "build_advisory", return_value={"advisory_guidance": {"status": "unavailable", "entries": []}, "sources": [], "safety_disclaimer": "test"}):
        no_location_response = client.post("/assess", data=submission("none", latitude="", longitude="", at_crop_location_confirmed="false"), content_type="multipart/form-data")
        no_location_weather_calls = weather_request.call_count
        manual_response = client.post("/assess", data=submission("manual", latitude="17.3850", longitude="78.4867", at_crop_location_confirmed="false"), content_type="multipart/form-data")
        auto_response = client.post("/assess", data=submission("auto", latitude="17.3850", longitude="78.4867", at_crop_location_confirmed="true"), content_type="multipart/form-data")
        model_calls_before_invalid = app.predict.load_prediction_artifacts.call_count
        xai_calls_before_invalid = app.xai.generate_gradcam_evidence.call_count
        fusion_calls_before_invalid = app.fusion.fuse_evidence.call_count
        advisory_calls_before_invalid = app.advisory.build_advisory.call_count
        invalid_manual_response = client.post("/assess", data=submission("manual", latitude="91", longitude="78.4867", at_crop_location_confirmed="false"), content_type="multipart/form-data")
        unconfirmed_auto_response = client.post("/assess", data=submission("auto", latitude="17.3850", longitude="78.4867", at_crop_location_confirmed="false"), content_type="multipart/form-data")
        invalid_submissions_stop_before_expensive_pipeline = (
            app.predict.load_prediction_artifacts.call_count == model_calls_before_invalid
            and app.xai.generate_gradcam_evidence.call_count == xai_calls_before_invalid
            and app.fusion.fuse_evidence.call_count == fusion_calls_before_invalid
            and app.advisory.build_advisory.call_count == advisory_calls_before_invalid
        )
    no_location_context = build_context.call_args_list[0].args[2]
    manual_context = build_context.call_args_list[1].args[2]
    auto_context = build_context.call_args_list[2].args[2]
    after = hashes()
    checks = {
        "page_shows_exactly_three_farmer_location_choices": all(text in home_html for text in ("Auto Location", "Manual Location", "No Location")) and "device_gps" not in home_html and "user_selected" not in home_html and "location_source" not in home_html,
        "no_location_submits_explicit_unavailable_contract": no_location_response.status_code == 200 and no_location_context is None,
        "manual_location_normalizes_to_user_selected": manual_response.status_code == 200 and manual_context == {"source": "user_selected", "latitude": 17.385, "longitude": 78.4867},
        "confirmed_auto_location_normalizes_to_device_gps": auto_response.status_code == 200 and auto_context == {"source": "device_gps", "latitude": 17.385, "longitude": 78.4867, "at_crop_location_confirmed": True},
        "invalid_manual_coordinates_are_rejected_friendly": invalid_manual_response.status_code == 400 and "valid geographic coordinates" in invalid_manual_response.get_data(as_text=True),
        "auto_location_requires_confirmation": unconfirmed_auto_response.status_code == 400 and "Confirm that your current device location" in unconfirmed_auto_response.get_data(as_text=True),
        "no_empty_location_source_reaches_context_validation": no_location_response.status_code == 200 and "Invalid location source" not in no_location_response.get_data(as_text=True),
        "no_location_does_not_call_weather": no_location_weather_calls == 0,
        "invalid_location_stops_before_model_xai_fusion_advisory": invalid_submissions_stop_before_expensive_pipeline,
        "prediction_unchanged": validated == prediction,
        "advisory_module_is_used": app.advisory.build_advisory.__module__ == "advisory",
        "protected_artifacts_unchanged": before == after,
    }
    print(checks)
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
