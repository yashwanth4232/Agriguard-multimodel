"""Central configuration for AgriGuard Model 1 (PlantVillage + MobileNetV2)."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths (portable, relative to repository root)
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
DATA_METADATA_DIR: Path = PROJECT_ROOT / "data" / "metadata"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_FIGURES_DIR: Path = PROJECT_ROOT / "reports" / "figures"
EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"

# ---------------------------------------------------------------------------
# Dataset split artifacts (M1-004)
# ---------------------------------------------------------------------------

COLOR_DATA_DIR: Path = DATA_RAW_DIR / "PlantVillage-Dataset" / "raw" / "color"
SPLIT_ASSIGNMENTS_PATH: Path = DATA_PROCESSED_DIR / "split_assignments.csv"
CLASS_TO_INDEX_PATH: Path = DATA_PROCESSED_DIR / "class_to_index.json"
SPLIT_METADATA_PATH: Path = DATA_METADATA_DIR / "dataset_split.json"
PLANTVILLAGE_METADATA_PATH: Path = DATA_METADATA_DIR / "plantvillage_metadata.json"

# ---------------------------------------------------------------------------
# Image and training defaults (COMMANDO.md §§4–6)
# ---------------------------------------------------------------------------

IMAGE_SIZE: tuple[int, int] = (224, 224)
IMAGE_CHANNELS: int = 3

RANDOM_SEED: int = 42

TRAIN_SPLIT: float = 0.70
VAL_SPLIT: float = 0.15
TEST_SPLIT: float = 0.15

BATCH_SIZE: int = 32
BASELINE_EPOCHS: int = 20
FINE_TUNE_EPOCHS: int = 10
BASELINE_LEARNING_RATE: float = 1e-3
FINE_TUNE_LEARNING_RATE: float = 1e-5

# ---------------------------------------------------------------------------
# Saved artifact paths (COMMANDO.md §11)
# ---------------------------------------------------------------------------

MODEL_PATH: Path = MODELS_DIR / "agriguard_mobilenetv2.keras"
CLASS_NAMES_PATH: Path = MODELS_DIR / "class_names.json"
OOD_REFERENCE_PATH: Path = MODELS_DIR / "ood_gap_reference.npz"
OOD_REFERENCE_METADATA_PATH: Path = MODELS_DIR / "ood_gap_reference.json"

TRAINING_HISTORY_PATH: Path = EXPERIMENTS_DIR / "training_history.csv"
EXPERIMENT_CONFIG_PATH: Path = EXPERIMENTS_DIR / "experiment_config.json"
TEST_METRICS_PATH: Path = EXPERIMENTS_DIR / "test_metrics.json"
CLASSIFICATION_REPORT_PATH: Path = EXPERIMENTS_DIR / "classification_report.csv"
CONFUSION_MATRIX_CSV_PATH: Path = EXPERIMENTS_DIR / "confusion_matrix.csv"

TRAINING_ACCURACY_FIG_PATH: Path = REPORTS_FIGURES_DIR / "training_accuracy.png"
TRAINING_LOSS_FIG_PATH: Path = REPORTS_FIGURES_DIR / "training_loss.png"
CONFUSION_MATRIX_FIG_PATH: Path = REPORTS_FIGURES_DIR / "confusion_matrix.png"
