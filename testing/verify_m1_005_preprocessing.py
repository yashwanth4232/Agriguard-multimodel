"""M1-005 verification for image loading and preprocessing pipeline."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config  # noqa: E402
import dataset  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_class_mapping() -> dict[str, object]:
    mapping = dataset.load_class_to_index()
    indices = sorted(mapping.values())
    return {
        "class_count": len(mapping),
        "expected_class_count": 38,
        "class_count_pass": len(mapping) == 38,
        "index_range_min": min(indices),
        "index_range_max": max(indices),
        "labels_within_mapping_pass": min(indices) == 0 and max(indices) == 37,
    }


def verify_dataset_shapes(split: str, tf_dataset: tf.data.Dataset) -> dict[str, object]:
    images, labels = next(iter(tf_dataset.batch(8)))
    shapes_ok = images.shape[1:] == (224, 224, 3)
    labels_list = labels.numpy().tolist()
    labels_ok = all(0 <= label <= 37 for label in labels_list)
    return {
        "split": split,
        "batch_image_shape": list(images.shape),
        "single_image_shape": list(images.shape[1:]),
        "shape_pass": shapes_ok,
        "labels_sample": labels_list,
        "labels_pass": labels_ok,
        "channels": int(images.shape[-1]),
    }


def verify_rgb_conversion() -> dict[str, object]:
    records = dataset.get_split_records("validation")
    sample_path = config.PROJECT_ROOT / records[0]["relative_path"]
    array = dataset._load_rgb_image_array(str(sample_path).encode("utf-8"))
    return {
        "sample_path": records[0]["relative_path"],
        "loaded_shape_before_tf_resize": list(array.shape),
        "channels_before_resize": int(array.shape[-1]),
        "rgb_pass": array.shape[-1] == 3,
    }


def verify_no_augmentation(split: str) -> dict[str, object]:
    tf_dataset = dataset.create_tf_dataset(split, shuffle=False, augment=False)
    first_images, _ = next(iter(tf_dataset.batch(2)))
    second_images, _ = next(iter(tf_dataset.batch(2)))
    identical = np.allclose(first_images.numpy(), second_images.numpy())
    return {
        "split": split,
        "deterministic_pass": bool(identical),
    }


def verify_train_augmentation_differs() -> dict[str, object]:
    record = dataset.get_split_records("train")[0]
    path = str(config.PROJECT_ROOT / record["relative_path"])
    label = dataset.load_class_to_index()[record["class_name"]]
    samples = []
    for _ in range(6):
        image, _ = dataset._map_example(
            tf.constant(path),
            tf.constant(label, dtype=tf.int32),
            augment=True,
        )
        samples.append(image.numpy())
    identical_pairs = sum(
        1 for left, right in zip(samples, samples[1:]) if np.allclose(left, right)
    )
    return {
        "augmentation_samples_checked": len(samples),
        "identical_consecutive_pairs": identical_pairs,
        "augmentation_active_pass": identical_pairs < len(samples) - 1,
    }


def verify_membership() -> dict[str, object]:
    expected = dataset.get_split_counts()
    actual = {
        "train": len(dataset.get_split_records("train")),
        "validation": len(dataset.get_split_records("validation")),
        "test": len(dataset.get_split_records("test")),
    }
    return {
        "expected_counts": expected,
        "dataset_counts": actual,
        "membership_pass": expected == actual,
        "total_pass": sum(actual.values()) == 54305,
    }


def verify_artifacts_unchanged(
    split_hash_before: str,
    sample_raw_hash_before: str,
) -> dict[str, object]:
    split_hash_after = sha256_file(config.SPLIT_ASSIGNMENTS_PATH)
    sample_path = config.PROJECT_ROOT / dataset.get_split_records("train")[0]["relative_path"]
    sample_raw_hash_after = sha256_file(sample_path)
    return {
        "split_assignments_unchanged": split_hash_before == split_hash_after,
        "sample_raw_file_unchanged": sample_raw_hash_before == sample_raw_hash_after,
        "raw_and_split_immutable_pass": (
            split_hash_before == split_hash_after
            and sample_raw_hash_before == sample_raw_hash_after
        ),
    }


def main() -> None:
    split_hash_before = sha256_file(config.SPLIT_ASSIGNMENTS_PATH)
    sample_raw_path = config.PROJECT_ROOT / dataset.get_split_records("train")[0]["relative_path"]
    sample_raw_hash_before = sha256_file(sample_raw_path)

    train_ds = dataset.create_train_dataset(shuffle=False)
    val_ds = dataset.create_validation_dataset()
    test_ds = dataset.create_test_dataset()

    results = {
        "class_mapping": verify_class_mapping(),
        "train_shapes": verify_dataset_shapes("train", train_ds),
        "validation_shapes": verify_dataset_shapes("validation", val_ds),
        "test_shapes": verify_dataset_shapes("test", test_ds),
        "rgb_conversion": verify_rgb_conversion(),
        "validation_no_augmentation": verify_no_augmentation("validation"),
        "test_no_augmentation": verify_no_augmentation("test"),
        "train_augmentation": verify_train_augmentation_differs(),
        "membership": verify_membership(),
        "artifact_immutability": verify_artifacts_unchanged(
            split_hash_before,
            sample_raw_hash_before,
        ),
        "memory_efficient_loading": {
            "approach": "tf.data.Dataset.from_tensor_slices with lazy map/load, batch, and prefetch",
            "loads_full_dataset_into_memory": False,
        },
    }

    print(json.dumps(results, indent=2))

    checks = [
        results["class_mapping"]["class_count_pass"],
        results["class_mapping"]["labels_within_mapping_pass"],
        results["train_shapes"]["shape_pass"],
        results["validation_shapes"]["shape_pass"],
        results["test_shapes"]["shape_pass"],
        results["rgb_conversion"]["rgb_pass"],
        results["validation_no_augmentation"]["deterministic_pass"],
        results["test_no_augmentation"]["deterministic_pass"],
        results["train_augmentation"]["augmentation_active_pass"],
        results["membership"]["membership_pass"],
        results["membership"]["total_pass"],
        results["artifact_immutability"]["raw_and_split_immutable_pass"],
    ]
    if not all(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
