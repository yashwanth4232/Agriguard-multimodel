# AGRIGUARD — MODEL 1 COMMANDO.md

## 0. PURPOSE
This file is the single source of truth for Model 1 of AgriGuard.
Model 1 is the visual crop-disease classification component using PlantVillage leaf images and a MobileNetV2 CNN with transfer learning.
If this file conflicts with an AI suggestion, this file wins.

## 1. MODEL 1 SCOPE
Model 1 must:
1. Obtain and verify the selected PlantVillage dataset.
2. Create a reproducible train/validation/test protocol.
3. Load and preprocess leaf images.
4. Build MobileNetV2 using transfer learning.
5. Train a disease classifier.
6. Evaluate accuracy, precision, recall, F1-score and confusion matrix.
7. Save the trained model and exact class mapping.
8. Provide local single-image inference.
9. Save experiment information required for the research report.

Model 1 does NOT include advisory generation, weather, soil, RAG, chatbot, OCR, mobile/web deployment, cloud deployment, or unrelated modules.

## 2. TECHNOLOGY CONTRACT
- Python 3.10
- Cursor
- PlantVillage
- TensorFlow + tf.keras
- MobileNetV2
- Pillow
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Optional Seaborn
- Git
OpenCV is optional and must not be added unless a specific requirement needs it.
Use MobileNetV2 through tensorflow.keras.

## 3. DATASET RULES
Record the exact PlantVillage source/URL, access date, total images, class count, class names, images per class, formats, duplicate policy, split strategy, preprocessing size and random seed.
Do not mix mirrors without documenting it.
Do not add another dataset silently.
Do not fabricate or silently remove labels/classes.
The final research report must cite the actual source used.

## 4. DATA SPLIT
Default: 70% train / 15% validation / 15% test.
Use a fixed random seed.
Test data must never be used for training or model selection.
Do not augment validation/test data.
Report class imbalance rather than hiding it.

## 5. IMAGE PREPROCESSING
Target input: 224x224x3 RGB.
Resize to 224x224, convert to RGB, use MobileNetV2-compatible preprocessing, and apply augmentation only to training.
Inference preprocessing must match validation/test preprocessing.
Save the class-to-index mapping.

## 6. MODEL
Input -> MobileNetV2(ImageNet pretrained) -> GlobalAveragePooling2D -> Dropout -> Dense softmax.
Stage A: freeze the backbone and train the head.
Stage B: optional controlled fine-tuning of upper layers with a small learning rate.
Record trainable layers.

## 7. RESEARCH INTEGRITY
Never fabricate accuracy, precision, recall, F1, training time, image counts, confusion matrices, inference time or comparison results.
Never tune against the test set.
Do not report the best test result from repeated untracked experiments.
Do not claim MobileNetV2 is best without a controlled comparison.
Do not claim field performance from PlantVillage-only testing.

## 8. EXPLAINABILITY BOUNDARY
XAI is part of the wider AgriGuard direction, but the first baseline must work before Grad-CAM or other XAI methods are added.

## 9. PROJECT STRUCTURE
AgriGuard/
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
├── models/
├── reports/
│   └── figures/
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── experiments/
├── testing/
├── requirements.txt
├── .gitignore
├── README.md
└── COMMANDO.md

## 10. FILE RESPONSIBILITIES
config.py: paths, image size, batch size, seed, epochs, learning rates and artifact names.
dataset.py: discovery, validation, class mapping, splitting, preprocessing and training augmentation.
model.py: MobileNetV2 construction and freeze/fine-tune logic.
train.py: dataset loading, model training, history/model/config saving.
evaluate.py: held-out test evaluation, metrics, classification report and confusion matrix.
predict.py: single-image inference using the exact saved model and class mapping.

## 11. REQUIRED ARTIFACTS
models/agriguard_mobilenetv2.keras
models/class_names.json
experiments/training_history.csv
experiments/experiment_config.json
experiments/test_metrics.json
experiments/classification_report.csv
experiments/confusion_matrix.csv
reports/figures/training_accuracy.png
reports/figures/training_loss.png
reports/figures/confusion_matrix.png

Do not overwrite earlier experiments without preserving configuration.

## 12. TESTING
After every step: syntax check, run relevant script, inspect outputs, check shapes, check leakage, inspect warnings/errors.
No unit-test framework is required initially.

## 13. CODING RULES
Use type hints where practical, pathlib for paths, centralized seeds/configuration, small functions, no TODOs/placeholders, no silent exception swallowing, no absolute machine-specific paths, no unnecessary dependencies.

## 14. AI CODING TOOL RULE
Cursor may assist with coding, but generated code must be inspected and must follow this file. Never ask an AI tool to invent research results, citations, datasets or experimental claims.

## 15. SESSION INITIALIZATION PROMPT
Read COMMANDO.md completely before making any changes. Inspect the existing AgriGuard project structure and determine which Model 1 implementation steps are already complete. Do not regenerate, overwrite, refactor, or delete working code. Implement only the specific step I request. Follow the dataset rules, model architecture rules, reproducibility requirements, research-integrity rules, file responsibilities, and testing policy in COMMANDO.md exactly. Do not fabricate results, datasets, metrics, citations, or files. If an instruction conflicts with COMMANDO.md, stop and report the conflict instead of guessing. After implementation, verify the requested behavior and report exactly what files changed, what commands were run, and whether each success criterion passed.

## 16. FIX PROMPT
The project fails with the following error:
[PASTE THE COMPLETE ERROR]
Read COMMANDO.md first. Fix only this specific error. Do not rewrite working code, change the architecture, change the dataset, change the model configuration, or introduce unrelated dependencies. After fixing it, run the relevant verification command and report the result.

## 17. GOLDEN RULES
1. COMMANDO.md is the source of truth.
2. One implementation step at a time.
3. Generate -> verify -> test -> manual check -> proceed.
4. Never fabricate research results.
5. Never leak test data into training.
6. Keep preprocessing identical between training and inference.
7. Record every experimental configuration.
8. Focus on genuine implementation and human-authored research interpretation.
9. Do not add unnecessary technologies.
10. Do not claim real-world field performance from a controlled dataset alone.
