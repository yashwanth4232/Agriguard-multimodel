"""Two-stage MobileNetV2 training for AgriGuard Model 1."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import tensorflow as tf

import config
import dataset
import evaluate
import model


REQUIRED_ARTIFACTS = (
    config.MODEL_PATH,
    config.CLASS_NAMES_PATH,
    config.TRAINING_HISTORY_PATH,
    config.EXPERIMENT_CONFIG_PATH,
    config.TEST_METRICS_PATH,
    config.CLASSIFICATION_REPORT_PATH,
    config.CONFUSION_MATRIX_CSV_PATH,
    config.TRAINING_ACCURACY_FIG_PATH,
    config.TRAINING_LOSS_FIG_PATH,
    config.CONFUSION_MATRIX_FIG_PATH,
)
CHECKPOINT_PATH = config.PROJECT_ROOT / "tmp" / "m1_007_retry_best_validation.keras"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_output_artifacts_are_new() -> None:
    existing = [str(path) for path in REQUIRED_ARTIFACTS if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing M1-007 artifacts: " + ", ".join(existing)
        )
    if CHECKPOINT_PATH.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing selection checkpoint: {CHECKPOINT_PATH}"
        )


def set_reproducibility_seed() -> None:
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def compile_model(trained_model: tf.keras.Model, learning_rate: float) -> None:
    trained_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def constrain_dataset_resources(
    source_dataset: tf.data.Dataset,
) -> tf.data.Dataset:
    """Limit input-pipeline concurrency to fit the local CPU memory budget."""
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    options.threading.max_intra_op_parallelism = 1
    return source_dataset.with_options(options)


def history_rows(history: tf.keras.callbacks.History, stage: str, start_epoch: int) -> list[dict[str, object]]:
    row_count = len(history.history["loss"])
    return [
        {
            "global_epoch": start_epoch + offset,
            "stage_epoch": offset + 1,
            "stage": stage,
            "loss": float(history.history["loss"][offset]),
            "accuracy": float(history.history["accuracy"][offset]),
            "val_loss": float(history.history["val_loss"][offset]),
            "val_accuracy": float(history.history["val_accuracy"][offset]),
        }
        for offset in range(row_count)
    ]


def main() -> None:
    ensure_output_artifacts_are_new()
    set_reproducibility_seed()
    start_time = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    split_sha256_before = sha256_file(config.SPLIT_ASSIGNMENTS_PATH)
    class_mapping_sha256_before = sha256_file(config.CLASS_TO_INDEX_PATH)

    class_to_index = dataset.load_class_to_index()
    class_names = [name for name, _ in sorted(class_to_index.items(), key=lambda item: item[1])]
    split_counts = dataset.get_split_counts()
    train_dataset = constrain_dataset_resources(
        dataset.create_train_dataset(config.BATCH_SIZE, shuffle=True)
    )
    validation_dataset = constrain_dataset_resources(
        dataset.create_validation_dataset(config.BATCH_SIZE)
    )
    test_dataset = constrain_dataset_resources(
        dataset.create_test_dataset(config.BATCH_SIZE)
    )

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(CHECKPOINT_PATH),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    stage_a = model.build_stage_a_model()
    compile_model(stage_a.model, config.BASELINE_LEARNING_RATE)
    stage_a_history = stage_a.model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=config.BASELINE_EPOCHS,
        callbacks=[checkpoint],
        verbose=2,
    )

    fine_tune_at = model.enable_fine_tuning(stage_a.backbone)
    compile_model(stage_a.model, config.FINE_TUNE_LEARNING_RATE)
    stage_b_history = stage_a.model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=config.FINE_TUNE_EPOCHS,
        callbacks=[checkpoint],
        verbose=2,
    )

    history_frame = pd.DataFrame(
        history_rows(stage_a_history, "stage_a_frozen_backbone", 1)
        + history_rows(stage_b_history, "stage_b_fine_tuning", config.BASELINE_EPOCHS + 1)
    )
    selected_model = tf.keras.models.load_model(CHECKPOINT_PATH)
    metrics, report_frame, matrix_frame = evaluate.evaluate_held_out_test_set(
        selected_model, test_dataset, class_names
    )
    duration_seconds = time.perf_counter() - start_time

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    selected_model.save(config.MODEL_PATH)
    config.CLASS_NAMES_PATH.write_text(
        json.dumps(class_names, indent=2) + "\n", encoding="utf-8"
    )
    history_frame.to_csv(config.TRAINING_HISTORY_PATH, index=False)
    evaluate.save_evaluation_artifacts(metrics, report_frame, matrix_frame)
    evaluate.save_training_figures(history_frame)
    evaluate.save_confusion_matrix_figure(matrix_frame)

    best_row = history_frame.loc[history_frame["val_accuracy"].idxmax()]
    experiment_config = {
        "run_started_at_utc": started_at,
        "training_duration_seconds": duration_seconds,
        "random_seed": config.RANDOM_SEED,
        "image_size": [*config.IMAGE_SIZE, config.IMAGE_CHANNELS],
        "batch_size": config.BATCH_SIZE,
        "num_classes": len(class_names),
        "optimizer": "Adam",
        "loss": "SparseCategoricalCrossentropy",
        "training_metric": "accuracy",
        "callbacks": [{"name": "ModelCheckpoint", "monitor": "val_accuracy", "mode": "max", "save_best_only": True}],
        "resource_controls": {
            "inter_op_parallelism_threads": 2,
            "intra_op_parallelism_threads": 2,
            "dataset_private_threadpool_size": 2,
            "dataset_max_intra_op_parallelism": 1,
        },
        "stage_a": {
            "epochs": config.BASELINE_EPOCHS,
            "learning_rate": config.BASELINE_LEARNING_RATE,
            "backbone": "MobileNetV2 ImageNet pretrained",
            "backbone_frozen": True,
            "trainable_layer_names": list(stage_a.trainable_layer_names),
        },
        "stage_b": {
            "epochs": config.FINE_TUNE_EPOCHS,
            "learning_rate": config.FINE_TUNE_LEARNING_RATE,
            "fine_tune_at": fine_tune_at,
            "unfrozen_backbone_layers": len(stage_a.backbone.layers) - fine_tune_at,
            "trainable_layer_names": list(model.get_trainable_layer_names(stage_a.model)),
        },
        "selection": {
            "split": "validation",
            "criterion": "maximum val_accuracy across Stage A and Stage B",
            "selected_global_epoch": int(best_row["global_epoch"]),
            "selected_stage": str(best_row["stage"]),
            "selected_validation_accuracy": float(best_row["val_accuracy"]),
        },
        "evaluation": {"split": "test", "performed_once_after_selection": True},
        "split_counts": split_counts,
        "split_assignments_sha256_before_training": split_sha256_before,
        "class_to_index_sha256_before_training": class_mapping_sha256_before,
        "augmentation": "Training split only; validation and test augmentation disabled.",
        "preprocessing": "RGB resize to 224x224 followed by MobileNetV2 preprocess_input.",
    }
    config.EXPERIMENT_CONFIG_PATH.write_text(
        json.dumps(experiment_config, indent=2) + "\n", encoding="utf-8"
    )
    CHECKPOINT_PATH.unlink()
    print(json.dumps({"training_duration_seconds": duration_seconds, "test_metrics": metrics, "selection": experiment_config["selection"]}, indent=2))


if __name__ == "__main__":
    main()
