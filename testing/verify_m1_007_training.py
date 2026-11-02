"""Verification for actual M1-007 training artifacts."""

from __future__ import annotations

import hashlib
import json
import sys

import pandas as pd
import tensorflow as tf

sys.path.insert(0, "src")

import config  # noqa: E402
import dataset  # noqa: E402


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    required = [
        config.MODEL_PATH, config.CLASS_NAMES_PATH, config.TRAINING_HISTORY_PATH,
        config.EXPERIMENT_CONFIG_PATH, config.TEST_METRICS_PATH,
        config.CLASSIFICATION_REPORT_PATH, config.CONFUSION_MATRIX_CSV_PATH,
        config.TRAINING_ACCURACY_FIG_PATH, config.TRAINING_LOSS_FIG_PATH,
        config.CONFUSION_MATRIX_FIG_PATH,
    ]
    artifacts_present = all(path.is_file() and path.stat().st_size > 0 for path in required)
    history = pd.read_csv(config.TRAINING_HISTORY_PATH)
    experiment_config = json.loads(config.EXPERIMENT_CONFIG_PATH.read_text(encoding="utf-8"))
    metrics = json.loads(config.TEST_METRICS_PATH.read_text(encoding="utf-8"))
    trained_model = tf.keras.models.load_model(config.MODEL_PATH)
    class_names = json.loads(config.CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    split_counts = dataset.get_split_counts()
    checks = {
        "artifacts_present": artifacts_present,
        "history_rows_pass": len(history) == config.BASELINE_EPOCHS + config.FINE_TUNE_EPOCHS,
        "stage_order_pass": history["stage"].tolist() == (["stage_a_frozen_backbone"] * config.BASELINE_EPOCHS + ["stage_b_fine_tuning"] * config.FINE_TUNE_EPOCHS),
        "model_output_pass": trained_model.output_shape == (None, 38),
        "class_names_pass": len(class_names) == 38,
        "metrics_pass": all(key in metrics for key in ("accuracy", "precision_weighted", "recall_weighted", "f1_score_weighted")),
        "selection_uses_validation_pass": experiment_config["selection"]["split"] == "validation",
        "test_evaluation_pass": experiment_config["evaluation"] == {"split": "test", "performed_once_after_selection": True},
        "split_counts_pass": split_counts == experiment_config["split_counts"],
        "split_immutable_pass": sha256_file(config.SPLIT_ASSIGNMENTS_PATH) == experiment_config["split_assignments_sha256_before_training"],
        "class_mapping_immutable_pass": sha256_file(config.CLASS_TO_INDEX_PATH) == experiment_config["class_to_index_sha256_before_training"],
    }
    print(json.dumps(checks, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
