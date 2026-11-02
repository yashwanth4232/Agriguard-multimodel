"""Read-only feature-space input-domain validation for AgriGuard Model 1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

import config
import predict

FEATURE_LAYER_NAME = "global_average_pooling2d"


class OODReferenceError(RuntimeError):
    """Raised when the separately versioned domain-reference artifact is invalid."""


def load_reference(path: Path = config.OOD_REFERENCE_PATH) -> dict[str, np.ndarray | float]:
    if not path.is_file():
        raise OODReferenceError(f"OOD reference artifact not found: {path}")
    with np.load(path, allow_pickle=False) as values:
        required = {"centroids", "feature_variance", "threshold"}
        if not required.issubset(values.files):
            raise OODReferenceError("OOD reference artifact is missing required arrays.")
        centroids = values["centroids"].astype(np.float32)
        variance = values["feature_variance"].astype(np.float32)
        threshold = float(values["threshold"])
    if centroids.shape != (38, 1280) or variance.shape != (1280,) or not np.isfinite(threshold):
        raise OODReferenceError("OOD reference artifact has an unexpected feature shape or threshold.")
    if not np.isfinite(centroids).all() or not np.isfinite(variance).all() or np.any(variance <= 0):
        raise OODReferenceError("OOD reference artifact contains invalid statistics.")
    return {"centroids": centroids, "feature_variance": variance, "threshold": threshold}


def make_feature_extractor(model: tf.keras.Model) -> tf.keras.Model:
    """Expose the saved model's deterministic pre-dropout 1,280-D GAP feature."""
    try:
        layer = model.get_layer(FEATURE_LAYER_NAME)
    except ValueError as exc:
        raise OODReferenceError(f"Model feature layer is unavailable: {FEATURE_LAYER_NAME}") from exc
    if tuple(layer.output.shape.as_list()) != (None, 1280):
        raise OODReferenceError("Model GAP feature is not the expected 1,280-dimensional representation.")
    return tf.keras.Model(model.inputs, layer.output, name="agriguard_gap_feature_extractor")


def domain_result(image_path: str | Path, model: tf.keras.Model) -> dict[str, object]:
    """Accept only images close to a PlantVillage class centroid in GAP space.

    This invokes the fixed backbone only.  It deliberately does not call the
    softmax classifier and therefore produces no disease label for rejection.
    """
    reference = load_reference()
    feature_model = make_feature_extractor(model)
    feature = feature_model(predict.preprocess_image_for_inference(Path(image_path)), training=False).numpy()[0]
    standardized = (reference["centroids"] - feature) ** 2 / reference["feature_variance"]
    score = float(np.min(np.mean(standardized, axis=1)))
    accepted = score <= float(reference["threshold"])
    return {
        "accepted": accepted,
        "status": "accepted" if accepted else "out_of_domain",
        "score": score,
        "threshold": float(reference["threshold"]),
        "feature_layer": FEATURE_LAYER_NAME,
    }
