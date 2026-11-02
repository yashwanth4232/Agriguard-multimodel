"""Verification for MobileNetV2 model construction (post M1-005)."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["KERAS_HOME"] = str(PROJECT_ROOT / ".keras")
os.environ["TEMP"] = str(PROJECT_ROOT / "tmp")
os.environ["TMP"] = str(PROJECT_ROOT / "tmp")
(PROJECT_ROOT / ".keras" / "models").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "tmp").mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

import config  # noqa: E402
import model  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_model_build() -> dict[str, object]:
    build_result = model.build_stage_a_model()
    built_model = build_result.model
    input_shape = built_model.input_shape
    output_shape = built_model.output_shape

    sample_input = tf.zeros((2, *config.IMAGE_SIZE, config.IMAGE_CHANNELS), dtype=tf.float32)
    sample_output = built_model(sample_input, training=False)

    backbone_layers = [layer.name for layer in build_result.backbone.layers]
    mobilenetv2_landmark_layers = {
        "Conv1",
        "block_1_expand",
        "block_1_depthwise",
        "block_16_project",
        "Conv_1",
        "out_relu",
    }
    mobilenetv2_backbone_present = mobilenetv2_landmark_layers.issubset(
        backbone_layers
    )
    backbone_trainable = sum(
        1 for layer in build_result.backbone.layers if layer.trainable
    )
    head_trainable = [
        name
        for name in build_result.trainable_layer_names
        if name in {"global_average_pooling2d", "dropout", "predictions"}
    ]
    param_counts = model.count_model_parameters(built_model)

    return {
        "model_built": True,
        "input_shape": list(input_shape),
        "input_shape_pass": input_shape == (None, *config.IMAGE_SIZE, config.IMAGE_CHANNELS),
        "output_shape": list(output_shape),
        "output_dimension_pass": output_shape[-1] == 38,
        "forward_output_shape": list(sample_output.shape),
        "forward_output_pass": sample_output.shape == (2, 38),
        "mobilenetv2_backbone_present": mobilenetv2_backbone_present,
        "backbone_layer_names_sample": backbone_layers[:3],
        "stage": build_result.stage,
        "backbone_trainable_layers": backbone_trainable,
        "backbone_frozen_pass": backbone_trainable == 0,
        "head_trainable_layers": head_trainable,
        "head_trainable_pass": head_trainable == [
            "global_average_pooling2d",
            "dropout",
            "predictions",
        ],
        "parameter_counts": param_counts,
        "training_performed": False,
    }


def verify_stage_b_configuration() -> dict[str, object]:
    build_result = model.build_stage_b_model()
    backbone_trainable = sum(
        1 for layer in build_result.backbone.layers if layer.trainable
    )
    return {
        "stage": build_result.stage,
        "fine_tune_at": build_result.fine_tune_at,
        "backbone_trainable_layers": backbone_trainable,
        "backbone_partially_trainable_pass": backbone_trainable > 0,
        "trainable_layer_count": len(build_result.trainable_layer_names),
    }


def verify_artifacts_unchanged(
    split_hash_before: str,
    class_map_hash_before: str,
) -> dict[str, object]:
    split_hash_after = sha256_file(config.SPLIT_ASSIGNMENTS_PATH)
    class_map_hash_after = sha256_file(config.CLASS_TO_INDEX_PATH)
    return {
        "split_assignments_unchanged": split_hash_before == split_hash_after,
        "class_mapping_unchanged": class_map_hash_before == class_map_hash_after,
        "artifacts_immutable_pass": (
            split_hash_before == split_hash_after
            and class_map_hash_before == class_map_hash_after
        ),
    }


def main() -> None:
    split_hash_before = sha256_file(config.SPLIT_ASSIGNMENTS_PATH)
    class_map_hash_before = sha256_file(config.CLASS_TO_INDEX_PATH)

    results = {
        "stage_a_model": verify_model_build(),
        "stage_b_model": verify_stage_b_configuration(),
        "artifact_immutability": verify_artifacts_unchanged(
            split_hash_before,
            class_map_hash_before,
        ),
    }

    print(json.dumps(results, indent=2))

    checks = [
        results["stage_a_model"]["input_shape_pass"],
        results["stage_a_model"]["output_dimension_pass"],
        results["stage_a_model"]["forward_output_pass"],
        results["stage_a_model"]["mobilenetv2_backbone_present"],
        results["stage_a_model"]["backbone_frozen_pass"],
        results["stage_a_model"]["head_trainable_pass"],
        results["stage_b_model"]["backbone_partially_trainable_pass"],
        results["artifact_immutability"]["artifacts_immutable_pass"],
    ]
    if not all(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
