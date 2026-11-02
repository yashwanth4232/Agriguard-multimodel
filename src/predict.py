"""Local single-image inference using the saved AgriGuard Model 1 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

import config
import dataset


def load_prediction_artifacts() -> tuple[tf.keras.Model, list[str]]:
    """Load the exact trained model and saved class-name order from M1-007."""
    if not config.MODEL_PATH.is_file():
        raise FileNotFoundError(f"Saved model not found: {config.MODEL_PATH}")
    if not config.CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(f"Saved class names not found: {config.CLASS_NAMES_PATH}")

    class_names = json.loads(config.CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    if not isinstance(class_names, list) or len(class_names) != 38:
        raise ValueError(
            f"Expected 38 saved class names in {config.CLASS_NAMES_PATH}, found {len(class_names)}."
        )
    return tf.keras.models.load_model(config.MODEL_PATH), class_names


def validate_image_path(image_path: Path) -> Path:
    """Confirm that the supplied path is an accessible, valid image file."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    try:
        with Image.open(image_path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Unreadable or invalid image file: {image_path}") from exc
    return image_path


def preprocess_image_for_inference(image_path: Path) -> tf.Tensor:
    """Apply the exact validation/test preprocessing contract to one image."""
    validated_path = validate_image_path(image_path)
    image = dataset._preprocess_image(
        tf.constant(str(validated_path)), augment=False
    )
    return tf.expand_dims(image, axis=0)


def predict_image(
    image_path: str | Path,
    trained_model: tf.keras.Model | None = None,
    class_names: list[str] | None = None,
) -> dict[str, object]:
    """Predict one local image and return its label and confidence."""
    path = Path(image_path)
    if trained_model is None or class_names is None:
        trained_model, class_names = load_prediction_artifacts()

    batch = preprocess_image_for_inference(path)
    probabilities = trained_model(batch, training=False).numpy()[0]
    if probabilities.shape != (len(class_names),):
        raise ValueError(
            f"Expected {len(class_names)} prediction scores, found {probabilities.shape}."
        )
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])
    return {
        "image_path": str(path),
        "predicted_index": predicted_index,
        "predicted_label": class_names[predicted_index],
        "confidence": confidence,
        "output_dimension": int(len(probabilities)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AgriGuard Model 1 inference on one local image."
    )
    parser.add_argument("image_path", type=Path, help="Path to one local image file.")
    return parser.parse_args()


def main() -> None:
    result = predict_image(parse_args().image_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
