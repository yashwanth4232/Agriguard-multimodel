"""Build the separate, read-only GAP-domain reference; never retrains Model 1."""

from __future__ import annotations

import hashlib, json
from pathlib import Path

import numpy as np

import config, dataset, ood, predict

SAMPLES_PER_CLASS = 50
VALIDATION_SAMPLES_PER_CLASS = 50
SEED = 20260816


def _feature_rows(records, feature_model, per_class):
    chosen, counts = [], {}
    for row in records:
        label = dataset.load_class_to_index()[row["class_name"]]
        if counts.get(label, 0) < per_class:
            chosen.append((label, config.PROJECT_ROOT / row["relative_path"]))
            counts[label] = counts.get(label, 0) + 1
    vectors, labels = [], []
    for label, path in chosen:
        vectors.append(feature_model(predict.preprocess_image_for_inference(path), training=False).numpy()[0])
        labels.append(label)
    return np.asarray(vectors, np.float32), np.asarray(labels, np.int32)


def main() -> None:
    model, _ = predict.load_prediction_artifacts()
    extractor = ood.make_feature_extractor(model)
    train_features, train_labels = _feature_rows(dataset.get_split_records("train"), extractor, SAMPLES_PER_CLASS)
    centroids = np.stack([train_features[train_labels == index].mean(axis=0) for index in range(38)]).astype(np.float32)
    variance = np.maximum(train_features.var(axis=0), 1e-6).astype(np.float32)
    validation, _ = _feature_rows(dataset.get_split_records("validation"), extractor, VALIDATION_SAMPLES_PER_CLASS)
    scores = np.min(np.mean((validation[:, None, :] - centroids[None, :, :]) ** 2 / variance[None, None, :], axis=2), axis=1)
    threshold = float(np.quantile(scores, 0.99))
    np.savez_compressed(config.OOD_REFERENCE_PATH, centroids=centroids, feature_variance=variance, threshold=np.asarray(threshold, np.float32))
    metadata = {"method":"minimum diagonal-Mahalanobis distance to 38 training-class GAP centroids","feature_layer":ood.FEATURE_LAYER_NAME,"feature_dimension":1280,"train_samples_per_class":SAMPLES_PER_CLASS,"validation_samples_per_class":VALIDATION_SAMPLES_PER_CLASS,"threshold_quantile":0.99,"threshold":threshold,"selection_note":"Validation data selects a 1% in-domain rejection operating point; no external OOD benchmark claim.","model_sha256":hashlib.sha256(config.MODEL_PATH.read_bytes()).hexdigest(),"split_sha256":hashlib.sha256(config.SPLIT_ASSIGNMENTS_PATH.read_bytes()).hexdigest(),"seed":SEED}
    config.OOD_REFERENCE_METADATA_PATH.write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")

if __name__ == "__main__": main()
