"""Read-only Grad-CAM explanations for AgriGuard Model 1 predictions."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import matplotlib
import matplotlib.cm as cm
import numpy as np
import tensorflow as tf
from PIL import Image

import config
import predict


TARGET_LAYER_NAME = "out_relu"
COLORMAP_NAME = "jet"
OVERLAY_ALPHA = 0.40
PREPROCESSING_DESCRIPTION = (
    "Pillow validation and RGB conversion; bilinear resize to 224x224; float32; "
    "tensorflow.keras.applications.mobilenet_v2.preprocess_input; batch dimension added "
    "through predict.preprocess_image_for_inference."
)
LIMITATIONS = [
    "Grad-CAM is an attribution visualization, not proof of biological causation.",
    "The 7x7 feature map is spatially coarse.",
    "Resizing improves visualization but does not improve true localization resolution.",
    "Heatmaps can highlight background or dataset-correlated features.",
    "The explanation is for the selected Model 1 class only.",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_display_image(image_path: str | Path) -> np.ndarray:
    """Load a separate unnormalized RGB image for visualization."""
    validated_path = predict.validate_image_path(Path(image_path))
    with Image.open(validated_path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def make_gradcam_heatmap(
    image_batch: tf.Tensor,
    model: tf.keras.Model,
    *,
    target_layer_name: str = TARGET_LAYER_NAME,
) -> tuple[np.ndarray, int, tuple[int | None, ...]]:
    """Return a normalized Grad-CAM heatmap for the model's own predicted class."""
    try:
        target_layer = model.get_layer(target_layer_name)
    except ValueError as exc:
        raise ValueError(f"Grad-CAM target layer not found: {target_layer_name}") from exc
    target_shape = tuple(target_layer.output.shape.as_list())
    if len(target_shape) != 4 or target_shape[1:3] != (7, 7):
        raise ValueError(
            f"Grad-CAM target layer must expose a 7x7 spatial feature map; got {target_shape}."
        )
    gradient_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output],
    )
    with tf.GradientTape() as tape:
        feature_maps, predictions = gradient_model(image_batch, training=False)
        class_index = tf.argmax(predictions[0], output_type=tf.int32)
        class_score = predictions[:, class_index]
    gradients = tape.gradient(class_score, feature_maps)
    if gradients is None:
        raise RuntimeError("Grad-CAM gradients were unavailable for the target layer.")
    channel_weights = tf.reduce_mean(gradients[0], axis=(0, 1))
    heatmap = tf.reduce_sum(feature_maps[0] * channel_weights, axis=-1)
    heatmap = tf.nn.relu(heatmap)
    maximum = tf.reduce_max(heatmap)
    normalized = tf.where(maximum > 0, heatmap / maximum, tf.zeros_like(heatmap))
    result = normalized.numpy().astype(np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("Grad-CAM heatmap contains non-finite values.")
    return result, int(class_index.numpy()), target_shape


def resize_heatmap(heatmap: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Resize a 2D heatmap to ``(height, width)`` using bilinear interpolation."""
    if heatmap.ndim != 2:
        raise ValueError("Grad-CAM heatmap must be two-dimensional.")
    height, width = image_size
    resized = tf.image.resize(heatmap[np.newaxis, ..., np.newaxis], (height, width), method="bilinear")
    return resized.numpy()[0, :, :, 0].astype(np.float32)


def create_overlay(display_image: np.ndarray, resized_heatmap: np.ndarray) -> np.ndarray:
    """Alpha-blend documented colormap output over an original RGB image."""
    if display_image.shape[:2] != resized_heatmap.shape:
        raise ValueError("Display image and resized heatmap dimensions must match.")
    colored = cm.get_cmap(COLORMAP_NAME)(np.clip(resized_heatmap, 0.0, 1.0))[..., :3]
    overlay = (1.0 - OVERLAY_ALPHA) * (display_image.astype(np.float32) / 255.0) + OVERLAY_ALPHA * colored
    return np.clip(overlay * 255.0, 0, 255).astype(np.uint8)


def generate_gradcam_evidence(
    image_path: str | Path,
    *,
    output_directory: str | Path = "reports/xai",
) -> dict[str, object]:
    """Generate XAI-only Grad-CAM artifacts for one image without changing Model 1."""
    path = Path(image_path)
    model, class_names = predict.load_prediction_artifacts()
    prediction = predict.predict_image(path, model, class_names)
    image_batch = predict.preprocess_image_for_inference(path)
    display_image = load_display_image(path)
    heatmap, target_class_index, target_layer_shape = make_gradcam_heatmap(image_batch, model)
    if target_class_index != prediction["predicted_index"]:
        raise RuntimeError("Grad-CAM target class does not match the Model 1 predicted class.")
    resized = resize_heatmap(heatmap, display_image.shape[:2])
    overlay = create_overlay(display_image, resized)

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    raw_path = output_path / "gradcam_raw_heatmap.npy"
    resized_path = output_path / "gradcam_resized_heatmap.npy"
    overlay_path = output_path / "gradcam_overlay.png"
    np.save(raw_path, heatmap)
    np.save(resized_path, resized)
    Image.fromarray(overlay, mode="RGB").save(overlay_path, format="PNG")

    metadata = {
        "input_image_path": str(path),
        "input_image_sha256": sha256_file(path),
        "model1_prediction": prediction,
        "target_class": prediction["predicted_label"],
        "target_class_index": target_class_index,
        "target_layer_name": TARGET_LAYER_NAME,
        "target_layer_output_shape": list(target_layer_shape),
        "preprocessing": PREPROCESSING_DESCRIPTION,
        "original_image_dimensions": [int(display_image.shape[1]), int(display_image.shape[0])],
        "heatmap_dimensions": list(heatmap.shape),
        "resized_heatmap_dimensions": [int(resized.shape[1]), int(resized.shape[0])],
        "overlay_dimensions": [int(overlay.shape[1]), int(overlay.shape[0])],
        "normalization": "Divided by maximum only when maximum > 0; otherwise an all-zero heatmap is retained.",
        "colormap": COLORMAP_NAME,
        "overlay_alpha": OVERLAY_ALPHA,
        "versions": {
            "tensorflow": tf.__version__, "numpy": np.__version__,
            "Pillow": version("Pillow"), "matplotlib": matplotlib.__version__,
        },
        "model_sha256": sha256_file(config.MODEL_PATH),
        "class_mapping_sha256": sha256_file(config.CLASS_NAMES_PATH),
        "artifacts": {
            "raw_heatmap": str(raw_path), "resized_heatmap": str(resized_path), "overlay": str(overlay_path),
        },
        "limitations": LIMITATIONS,
    }
    return metadata


def write_xai_metadata(metadata: dict[str, object], path: str | Path = "reports/xai/xai_verification.json") -> None:
    """Write XAI-specific JSON metadata; no Model 1 artifact is overwritten."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
