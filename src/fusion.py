"""Deterministic, provenance-preserving association of AgriGuard evidence."""

from __future__ import annotations

from typing import Mapping

import context


FUSION_TYPE = "deterministic_context_association"
LIMITATIONS = [
    "Current weather is not image-capture-time weather.",
    "Context does not independently diagnose or confirm disease.",
]


class FusionValidationError(ValueError):
    """Raised when evidence sources do not describe the same Model 1 result."""


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise FusionValidationError(f"Invalid {name}: expected a mapping.")
    return value


def _require_equal(actual: object, expected: object, field: str) -> None:
    if actual != expected:
        raise FusionValidationError(f"Inconsistent {field}: expected {expected!r}, found {actual!r}.")


def build_fusion_input(
    model1_prediction: Mapping[str, object],
    normalized_context: Mapping[str, object],
    xai_metadata: Mapping[str, object] | None = None,
    *,
    model_artifact_sha256: str | None = None,
) -> dict[str, object]:
    """Validate and arrange source evidence without changing any diagnosis value."""
    prediction = context.validate_model1_prediction(model1_prediction)
    record = _mapping(normalized_context, "agricultural context")
    crop, disease = context.extract_crop_and_disease(str(prediction["predicted_label"]))
    _require_equal(record.get("crop"), crop, "context crop")
    _require_equal(record.get("disease"), disease, "context disease")
    _require_equal(record.get("model1_confidence"), prediction["confidence"], "context Model 1 confidence")

    xai_evidence: dict[str, object] | None = None
    if xai_metadata is not None:
        xai = _mapping(xai_metadata, "XAI evidence")
        _require_equal(xai.get("target_class"), prediction["predicted_label"], "XAI target class")
        _require_equal(xai.get("target_class_index"), prediction["predicted_index"], "XAI target class index")
        artifacts = xai.get("artifacts")
        artifact_map = artifacts if isinstance(artifacts, Mapping) else {}
        xai_evidence = {
            "target_class": xai["target_class"],
            "target_class_index": xai["target_class_index"],
            "target_layer_name": xai.get("target_layer_name"),
            "heatmap_reference": artifact_map.get("raw_heatmap"),
            "overlay_reference": artifact_map.get("overlay"),
        }
    return {
        "visual_evidence": {
            "predicted_index": prediction["predicted_index"],
            "predicted_label": prediction["predicted_label"],
            "crop": crop,
            "disease": disease,
            "confidence": prediction["confidence"],
            "model_artifact_sha256": model_artifact_sha256,
        },
        "context_evidence": dict(record),
        "xai_evidence": xai_evidence,
    }


def fuse_evidence(
    model1_prediction: Mapping[str, object],
    normalized_context: Mapping[str, object],
    xai_metadata: Mapping[str, object] | None = None,
    *,
    model_artifact_sha256: str | None = None,
) -> dict[str, object]:
    """Return an advisory-ready record; no score, disease, or context is invented."""
    fusion_input = build_fusion_input(
        model1_prediction, normalized_context, xai_metadata,
        model_artifact_sha256=model_artifact_sha256,
    )
    visual = fusion_input["visual_evidence"]
    record = fusion_input["context_evidence"]
    xai = fusion_input["xai_evidence"]
    weather_available = record.get("weather_status") == "available"
    location_available = record.get("location_status") == "available"
    return {
        "fusion_type": FUSION_TYPE,
        "diagnosis": {
            "source": "Model 1", **visual, "unchanged_by_fusion": True,
        },
        "agricultural_context": record,
        "visual_explanation": (
            {"available": False, "source": None, "target_class": None,
             "target_class_index": None, "target_layer_name": None,
             "overlay_reference": None}
            if xai is None else {"available": True, "source": "Grad-CAM for Model 1", **xai}
        ),
        "availability": {
            "weather_available": weather_available,
            "location_available": location_available,
            "growth_stage_known": record.get("growth_stage") != "Unknown",
            "xai_available": xai is not None,
        },
        "advisory_inputs": {
            "disease_label": visual["predicted_label"], "crop": visual["crop"],
            "confidence": visual["confidence"], "growth_stage": record.get("growth_stage"),
            "current_weather_context_available": weather_available,
        },
        "provenance": {
            "model1": "src/predict.py", "context": "src/context.py",
            "farmer_input": "src/farmer_inputs.py", "weather_provider": record.get("weather_provider"),
            "xai": "src/xai.py" if xai is not None else None,
        },
        "limitations": LIMITATIONS,
    }
