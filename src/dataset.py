"""Dataset discovery and reproducible train/validation/test splitting for Model 1."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import config

SplitName = Literal["train", "validation", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class SplitUnit:
    unit_id: str
    class_name: str
    image_paths: tuple[Path, ...]
    duplicate_group: bool


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def discover_class_images(color_dir: Path) -> dict[str, list[Path]]:
    if not color_dir.is_dir():
        raise FileNotFoundError(f"Color dataset directory not found: {color_dir}")

    class_images: dict[str, list[Path]] = {}
    for class_dir in sorted(p for p in color_dir.iterdir() if p.is_dir()):
        paths = sorted(p for p in class_dir.iterdir() if is_image_file(p))
        class_images[class_dir.name] = paths
    return class_images


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_duplicate_groups(
    class_images: dict[str, list[Path]],
) -> dict[str, list[Path]]:
    hash_to_paths: dict[str, list[Path]] = defaultdict(list)
    for paths in class_images.values():
        for path in paths:
            hash_to_paths[sha256_file(path)].append(path)

    duplicate_groups: dict[str, list[Path]] = {}
    for content_hash, paths in hash_to_paths.items():
        if len(paths) > 1:
            group_id = f"dup_{content_hash[:16]}"
            duplicate_groups[group_id] = sorted(paths, key=lambda p: p.as_posix())
    return duplicate_groups


def build_split_units(class_images: dict[str, list[Path]]) -> list[SplitUnit]:
    duplicate_groups = build_duplicate_groups(class_images)
    grouped_paths = {path for paths in duplicate_groups.values() for path in paths}
    units: list[SplitUnit] = []

    for group_id, paths in sorted(duplicate_groups.items()):
        class_name = paths[0].parent.name
        units.append(
            SplitUnit(
                unit_id=group_id,
                class_name=class_name,
                image_paths=tuple(paths),
                duplicate_group=True,
            )
        )

    for class_name, paths in sorted(class_images.items()):
        for path in paths:
            if path in grouped_paths:
                continue
            units.append(
                SplitUnit(
                    unit_id=f"single_{path.stem}",
                    class_name=class_name,
                    image_paths=(path,),
                    duplicate_group=False,
                )
            )

    return units


def split_units_for_class(
    units: list[SplitUnit],
    random_seed: int,
) -> dict[SplitName, list[SplitUnit]]:
    if not units:
        raise ValueError("Cannot split an empty unit list.")

    if len(units) == 1:
        return {"train": units, "validation": [], "test": []}

    train_units, holdout_units = train_test_split(
        units,
        test_size=config.VAL_SPLIT + config.TEST_SPLIT,
        random_state=random_seed,
    )
    validation_units, test_units = train_test_split(
        holdout_units,
        test_size=config.TEST_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT),
        random_state=random_seed,
    )
    return {
        "train": train_units,
        "validation": validation_units,
        "test": test_units,
    }


def create_class_to_index(class_names: list[str]) -> dict[str, int]:
    return {name: index for index, name in enumerate(class_names)}


def relative_project_path(path: Path) -> str:
    return path.relative_to(config.PROJECT_ROOT).as_posix()


def create_split_assignments(
    random_seed: int | None = None,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    seed = config.RANDOM_SEED if random_seed is None else random_seed
    class_images = discover_class_images(config.COLOR_DATA_DIR)
    class_names = sorted(class_images.keys())

    units_by_class: dict[str, list[SplitUnit]] = defaultdict(list)
    for unit in build_split_units(class_images):
        units_by_class[unit.class_name].append(unit)

    split_units: dict[SplitName, list[SplitUnit]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for class_name in class_names:
        class_split = split_units_for_class(units_by_class[class_name], seed)
        for split_name in split_units:
            split_units[split_name].extend(class_split[split_name])

    rows: list[dict[str, str]] = []
    for split_name, units in split_units.items():
        for unit in units:
            for path in unit.image_paths:
                rows.append(
                    {
                        "relative_path": relative_project_path(path),
                        "class_name": unit.class_name,
                        "split": split_name,
                        "unit_id": unit.unit_id,
                        "duplicate_group": str(unit.duplicate_group).lower(),
                    }
                )

    rows.sort(key=lambda row: row["relative_path"])

    metadata = build_split_metadata(
        rows=rows,
        class_names=class_names,
        random_seed=seed,
        duplicate_groups=build_duplicate_groups(class_images),
    )
    return rows, metadata


def build_split_metadata(
    rows: list[dict[str, str]],
    class_names: list[str],
    random_seed: int,
    duplicate_groups: dict[str, list[Path]],
) -> dict[str, object]:
    split_counts = {"train": 0, "validation": 0, "test": 0}
    per_class_counts: dict[str, dict[str, int]] = {
        class_name: {"train": 0, "validation": 0, "test": 0} for class_name in class_names
    }
    unit_split_map: dict[str, str] = {}

    for row in rows:
        split_name = row["split"]
        split_counts[split_name] += 1
        per_class_counts[row["class_name"]][split_name] += 1
        unit_split_map[row["unit_id"]] = split_name

    duplicate_group_handling = []
    leakage_groups = 0
    for group_id, paths in sorted(duplicate_groups.items()):
        splits = {row["split"] for row in rows if row["unit_id"] == group_id}
        if len(splits) > 1:
            leakage_groups += 1
        duplicate_group_handling.append(
            {
                "duplicate_group_id": group_id,
                "image_count": len(paths),
                "class_name": paths[0].parent.name,
                "assigned_split": sorted(splits)[0] if len(splits) == 1 else sorted(splits),
                "relative_paths": [relative_project_path(path) for path in paths],
            }
        )

    return {
        "source_url": "https://github.com/spMohanty/PlantVillage-Dataset",
        "selected_subset": "raw/color/",
        "local_dataset_path": relative_project_path(
            config.DATA_RAW_DIR / "PlantVillage-Dataset"
        ),
        "random_seed": random_seed,
        "split_ratios": {
            "train": config.TRAIN_SPLIT,
            "validation": config.VAL_SPLIT,
            "test": config.TEST_SPLIT,
        },
        "split_method": (
            "Class-wise stratified split at duplicate-aware unit level using "
            "sklearn.model_selection.train_test_split with random_state=42. "
            "Each exact duplicate group is treated as one indivisible unit."
        ),
        "leakage_prevention_method": (
            "Images are assigned to exactly one split. Exact duplicate groups "
            "identified by SHA-256 are kept intact within a single split. "
            "Validation and test sets are isolated from training/model selection."
        ),
        "duplicate_handling": (
            "All 21 exact duplicate groups are retained. Each duplicate group "
            "is assigned as a single unit to train, validation, or test so that "
            "identical files never appear in multiple splits."
        ),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_groups": duplicate_group_handling,
        "duplicate_group_leakage_count": leakage_groups,
        "total_images": sum(split_counts.values()),
        "split_counts": split_counts,
        "per_class_split_counts": per_class_counts,
        "class_count": len(class_names),
        "class_names": class_names,
        "artifacts": {
            "split_assignments_csv": relative_project_path(config.SPLIT_ASSIGNMENTS_PATH),
            "class_to_index_json": relative_project_path(config.CLASS_TO_INDEX_PATH),
        },
    }


def save_split_artifacts(
    rows: list[dict[str, str]],
    metadata: dict[str, object],
    class_names: list[str],
) -> None:
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.DATA_METADATA_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "relative_path",
        "class_name",
        "split",
        "unit_id",
        "duplicate_group",
    ]
    with config.SPLIT_ASSIGNMENTS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    class_to_index = create_class_to_index(class_names)
    config.CLASS_TO_INDEX_PATH.write_text(
        json.dumps(class_to_index, indent=2) + "\n",
        encoding="utf-8",
    )

    config.SPLIT_METADATA_PATH.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    plantvillage_metadata_path = config.PLANTVILLAGE_METADATA_PATH
    if plantvillage_metadata_path.exists():
        plantvillage_metadata = json.loads(
            plantvillage_metadata_path.read_text(encoding="utf-8")
        )
        plantvillage_metadata["split_strategy"] = (
            "70/15/15 train/validation/test split created in M1-004 using random "
            f"seed {config.RANDOM_SEED}; see data/metadata/dataset_split.json."
        )
        plantvillage_metadata["split_record"] = relative_project_path(
            config.SPLIT_METADATA_PATH
        )
        plantvillage_metadata_path.write_text(
            json.dumps(plantvillage_metadata, indent=2) + "\n",
            encoding="utf-8",
        )


def run_split(random_seed: int | None = None) -> dict[str, object]:
    rows, metadata = create_split_assignments(random_seed=random_seed)
    class_names = metadata["class_names"]
    save_split_artifacts(rows, metadata, class_names)
    return metadata


def load_split_assignments() -> list[dict[str, str]]:
    with config.SPLIT_ASSIGNMENTS_PATH.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_class_to_index() -> dict[str, int]:
    mapping = json.loads(config.CLASS_TO_INDEX_PATH.read_text(encoding="utf-8"))
    if len(mapping) != 38:
        raise ValueError(
            f"Expected 38 classes in {config.CLASS_TO_INDEX_PATH}, found {len(mapping)}."
        )
    return mapping


def load_index_to_class() -> dict[int, str]:
    class_to_index = load_class_to_index()
    return {index: name for name, index in class_to_index.items()}


def get_split_records(split: SplitName) -> list[dict[str, str]]:
    return [row for row in load_split_assignments() if row["split"] == split]


def get_split_counts() -> dict[str, int]:
    counts = {"train": 0, "validation": 0, "test": 0}
    for row in load_split_assignments():
        counts[row["split"]] += 1
    return counts


def _load_rgb_image_array(path_bytes: bytes | object) -> np.ndarray:
    if hasattr(path_bytes, "numpy"):
        path_bytes = path_bytes.numpy()
    if isinstance(path_bytes, bytes):
        path = path_bytes.decode("utf-8")
    else:
        path = str(path_bytes)
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            return np.array(rgb, dtype=np.uint8)
    except Exception as exc:
        raise RuntimeError(f"Cannot load image at path: {path}") from exc


def apply_train_augmentation(image: tf.Tensor) -> tf.Tensor:
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return tf.clip_by_value(image, 0.0, 255.0)


def _preprocess_image(path: tf.Tensor, augment: bool) -> tf.Tensor:
    image = tf.py_function(func=_load_rgb_image_array, inp=[path], Tout=tf.uint8)
    image.set_shape([None, None, config.IMAGE_CHANNELS])
    image = tf.image.resize(image, config.IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    image = tf.cast(image, tf.float32)
    if augment:
        image = apply_train_augmentation(image)
    image = preprocess_input(image)
    image.set_shape([config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], config.IMAGE_CHANNELS])
    return image


def _map_example(path: tf.Tensor, label: tf.Tensor, augment: bool) -> tuple[tf.Tensor, tf.Tensor]:
    image = _preprocess_image(path, augment=augment)
    return image, label


def create_tf_dataset(
    split: SplitName,
    batch_size: int | None = None,
    shuffle: bool = False,
    augment: bool | None = None,
) -> tf.data.Dataset:
    class_to_index = load_class_to_index()
    records = get_split_records(split)
    if not records:
        raise ValueError(f"No records found for split '{split}'.")

    paths = [str(config.PROJECT_ROOT / row["relative_path"]) for row in records]
    labels = [class_to_index[row["class_name"]] for row in records]

    use_augmentation = split == "train" if augment is None else augment
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=config.RANDOM_SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        lambda path, label: _map_example(path, label, augment=use_augmentation),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not use_augmentation,
    )
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    return dataset.prefetch(tf.data.AUTOTUNE)


def create_train_dataset(batch_size: int | None = None, shuffle: bool = True) -> tf.data.Dataset:
    return create_tf_dataset("train", batch_size=batch_size, shuffle=shuffle, augment=True)


def create_validation_dataset(batch_size: int | None = None) -> tf.data.Dataset:
    return create_tf_dataset("validation", batch_size=batch_size, shuffle=False, augment=False)


def create_test_dataset(batch_size: int | None = None) -> tf.data.Dataset:
    return create_tf_dataset("test", batch_size=batch_size, shuffle=False, augment=False)


if __name__ == "__main__":
    result = run_split()
    print(json.dumps(result["split_counts"], indent=2))
