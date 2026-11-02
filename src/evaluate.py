"""Held-out test evaluation and report generation for AgriGuard Model 1."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

import config


def evaluate_held_out_test_set(
    trained_model: tf.keras.Model,
    test_dataset: tf.data.Dataset,
    class_names: list[str],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Evaluate a selected model once on the held-out test dataset."""
    test_loss, test_accuracy = trained_model.evaluate(test_dataset, verbose=2)
    probabilities = trained_model.predict(test_dataset, verbose=2)
    true_labels = np.concatenate(
        [labels.numpy() for _, labels in test_dataset], axis=0
    )
    predicted_labels = np.argmax(probabilities, axis=1)
    labels = list(range(len(class_names)))
    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            labels=labels,
            average="weighted",
            zero_division=0,
        )
    )
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_frame = pd.DataFrame(report).transpose()
    matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)
    matrix_frame = pd.DataFrame(matrix, index=class_names, columns=class_names)
    metrics: dict[str, object] = {
        "loss": float(test_loss),
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "keras_accuracy": float(test_accuracy),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_score_weighted": float(f1_weighted),
        "test_examples": int(len(true_labels)),
        "evaluation_split": "test",
    }
    return metrics, report_frame, matrix_frame


def save_evaluation_artifacts(
    metrics: dict[str, object],
    report_frame: pd.DataFrame,
    matrix_frame: pd.DataFrame,
) -> None:
    """Persist actual test metrics, classification report, and confusion matrix."""
    config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with config.TEST_METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    report_frame.to_csv(config.CLASSIFICATION_REPORT_PATH, index=True)
    matrix_frame.to_csv(config.CONFUSION_MATRIX_CSV_PATH, index=True)


def save_training_figures(history_frame: pd.DataFrame) -> None:
    """Create training accuracy and loss plots from the recorded training history."""
    config.REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(history_frame["global_epoch"], history_frame["accuracy"], label="train")
    plt.plot(
        history_frame["global_epoch"], history_frame["val_accuracy"], label="validation"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.TRAINING_ACCURACY_FIG_PATH, dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(history_frame["global_epoch"], history_frame["loss"], label="train")
    plt.plot(history_frame["global_epoch"], history_frame["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.TRAINING_LOSS_FIG_PATH, dpi=150)
    plt.close()


def save_confusion_matrix_figure(matrix_frame: pd.DataFrame) -> None:
    """Create a labeled confusion-matrix figure from the held-out test result."""
    plt.figure(figsize=(18, 15))
    plt.imshow(matrix_frame.to_numpy(), interpolation="nearest", cmap="Blues")
    plt.title("Held-out test confusion matrix")
    plt.colorbar()
    tick_positions = np.arange(len(matrix_frame.index))
    plt.xticks(tick_positions, matrix_frame.columns, rotation=90, fontsize=6)
    plt.yticks(tick_positions, matrix_frame.index, fontsize=6)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.tight_layout()
    plt.savefig(config.CONFUSION_MATRIX_FIG_PATH, dpi=150)
    plt.close()
