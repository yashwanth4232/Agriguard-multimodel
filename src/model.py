"""MobileNetV2 construction and freeze/fine-tune logic for Model 1."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import MobileNetV2

import config
import dataset

DEFAULT_DROPOUT_RATE = 0.2
DEFAULT_FINE_TUNE_UNFROZEN_LAYERS = 30
IMAGENET_WEIGHTS_PATH = (
    config.PROJECT_ROOT
    / ".keras"
    / "models"
    / "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"
)


@dataclass(frozen=True)
class ModelBuildResult:
    model: Model
    backbone: MobileNetV2
    num_classes: int
    stage: str
    fine_tune_at: int | None
    trainable_layer_names: tuple[str, ...]


def get_num_classes() -> int:
    return len(dataset.load_class_to_index())


def build_mobilenetv2_model(
    num_classes: int | None = None,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
    weights: str = "imagenet",
) -> tuple[Model, MobileNetV2]:
    if num_classes is None:
        num_classes = get_num_classes()

    input_shape = (*config.IMAGE_SIZE, config.IMAGE_CHANNELS)
    if weights == "imagenet" and IMAGENET_WEIGHTS_PATH.is_file():
        backbone = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=None,
            pooling=None,
        )
        backbone.load_weights(str(IMAGENET_WEIGHTS_PATH))
    else:
        backbone = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=weights,
            pooling=None,
        )
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(backbone.output)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)
    model = Model(
        inputs=backbone.input,
        outputs=outputs,
        name="agriguard_mobilenetv2",
    )
    return model, backbone


def freeze_backbone(backbone: MobileNetV2) -> None:
    backbone.trainable = False


def enable_fine_tuning(
    backbone: MobileNetV2,
    fine_tune_at: int | None = None,
) -> int:
    if fine_tune_at is None:
        fine_tune_at = max(0, len(backbone.layers) - DEFAULT_FINE_TUNE_UNFROZEN_LAYERS)

    backbone.trainable = True
    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False
    return fine_tune_at


def get_trainable_layer_names(model: Model) -> tuple[str, ...]:
    return tuple(layer.name for layer in model.layers if layer.trainable)


def build_stage_a_model(
    num_classes: int | None = None,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
    weights: str = "imagenet",
) -> ModelBuildResult:
    model, backbone = build_mobilenetv2_model(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        weights=weights,
    )
    freeze_backbone(backbone)
    return ModelBuildResult(
        model=model,
        backbone=backbone,
        num_classes=num_classes or get_num_classes(),
        stage="stage_a_frozen_backbone",
        fine_tune_at=None,
        trainable_layer_names=get_trainable_layer_names(model),
    )


def build_stage_b_model(
    num_classes: int | None = None,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
    weights: str = "imagenet",
    fine_tune_at: int | None = None,
) -> ModelBuildResult:
    model, backbone = build_mobilenetv2_model(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        weights=weights,
    )
    resolved_fine_tune_at = enable_fine_tuning(backbone, fine_tune_at=fine_tune_at)
    return ModelBuildResult(
        model=model,
        backbone=backbone,
        num_classes=num_classes or get_num_classes(),
        stage="stage_b_fine_tuning",
        fine_tune_at=resolved_fine_tune_at,
        trainable_layer_names=get_trainable_layer_names(model),
    )


def count_model_parameters(model: Model) -> dict[str, int]:
    trainable = int(sum(tf.size(variable).numpy() for variable in model.trainable_variables))
    total = int(model.count_params())
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "non_trainable_parameters": total - trainable,
    }
