# AgriGuard

AgriGuard is a crop-disease detection and advisory system. This repository implements the project in phases; each phase has a defined scope and artifacts.

## Model 1 — Visual Crop-Disease Classification (Baseline)

Model 1 is the first component: a visual crop-disease classification baseline using the **PlantVillage** leaf-image dataset and a **MobileNetV2** convolutional neural network with transfer learning.

Model 1 covers dataset verification, reproducible train/validation/test splits, image preprocessing, model training, held-out evaluation (accuracy, precision, recall, F1, confusion matrix), saved model artifacts, and local single-image inference. It does not include advisory generation, weather, soil, RAG, chatbot, deployment, or other later-phase features.

All Model 1 requirements, architecture rules, and file responsibilities are defined in [`COMMANDO.md`](COMMANDO.md). That file is the single source of truth for Model 1 implementation.

## Project Layout

```
AgriGuard-Multimodal/
├── COMMANDO.md          # Model 1 specification (source of truth)
├── README.md
├── data/
│   ├── raw/             # Original PlantVillage data
│   ├── processed/       # Processed splits and derived data
│   └── metadata/        # Dataset documentation and manifests
├── models/              # Trained model and class mapping (generated)
├── reports/figures/     # Evaluation and training figures (generated)
├── src/                 # Python source (config, dataset, model, train, evaluate, predict)
├── experiments/         # Training history, metrics, experiment configs (generated)
└── testing/             # Verification scripts and checks
```

## Getting Started

Model 1 development follows the step-by-step plan in `COMMANDO.md`. Do not skip verification steps between implementations.
