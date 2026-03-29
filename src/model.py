"""InceptionV3 model architecture builder and training utilities."""

import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.metrics import Precision, Recall

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def build_model(
    num_classes: int = config.NUM_CLASSES,
    dropout_rate: float = config.DROPOUT_RATE,
    l2_reg: float = config.L2_REG,
    dense_units_1: int = config.DENSE_UNITS_1,
    dense_units_2: int = config.DENSE_UNITS_2,
) -> tuple[keras.Model, keras.Model]:
    """Build InceptionV3-based classification model.

    Architecture:
        InceptionV3 (ImageNet) -> GAP -> Dense(256)+BN+Dropout
        -> Dense(128)+Dropout -> Dense(4, softmax)

    Args:
        num_classes: Number of output classes.
        dropout_rate: Dropout rate for regularization.
        l2_reg: L2 regularization factor.
        dense_units_1: Units in first dense layer.
        dense_units_2: Units in second dense layer.

    Returns:
        Tuple of (full_model, base_model).
    """
    base_model = InceptionV3(
        weights="imagenet",
        include_top=False,
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
    )

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        dense_units_1,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(
        dense_units_2,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
    )(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=base_model.input, outputs=outputs)
    return model, base_model


def freeze_backbone(base_model: keras.Model) -> None:
    """Freeze all layers in the backbone."""
    for layer in base_model.layers:
        layer.trainable = False


def unfreeze_from(base_model: keras.Model, layer_name: str = "mixed7") -> None:
    """Unfreeze backbone layers from the specified layer onwards.

    Args:
        base_model: The InceptionV3 base model.
        layer_name: Name of the layer from which to start unfreezing.
    """
    unfreeze = False
    for layer in base_model.layers:
        if layer.name == layer_name:
            unfreeze = True
        layer.trainable = unfreeze

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    total_count = len(base_model.layers)
    print(f"Unfroze from '{layer_name}': {trainable_count}/{total_count} layers trainable")


def unfreeze_all(base_model: keras.Model) -> None:
    """Unfreeze all layers in the backbone."""
    for layer in base_model.layers:
        layer.trainable = True
    print(f"All {len(base_model.layers)} backbone layers unfrozen")


def compile_model(
    model: keras.Model,
    learning_rate: float,
    stage: int = 1,
) -> None:
    """Compile model with AdamW optimizer.

    Args:
        model: The Keras model to compile.
        learning_rate: Learning rate for the optimizer.
        stage: Training stage number (for logging).
    """
    optimizer = keras.optimizers.AdamW(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            Precision(name="precision"),
            Recall(name="recall"),
        ],
    )
    print(f"Stage {stage}: compiled with LR={learning_rate:.1e}, "
          f"trainable params: {model.count_params():,}")


def get_callbacks(
    output_dir: str = config.OUTPUT_DIR,
    stage: int = 1,
) -> list:
    """Create training callbacks.

    Args:
        output_dir: Base directory for saving outputs.
        stage: Training stage number.

    Returns:
        List of Keras callbacks.
    """
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=os.path.join(output_dir, "models", f"best_stage{stage}.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        TensorBoard(
            log_dir=os.path.join(output_dir, "logs", f"stage{stage}"),
        ),
        CSVLogger(
            os.path.join(output_dir, "logs", f"stage{stage}_metrics.csv"),
            separator=",",
            append=False,
        ),
    ]
    return callbacks


def build_tunable_model(hp) -> keras.Model:
    """Keras Tuner compatible model builder.

    Args:
        hp: HyperParameters object from Keras Tuner.

    Returns:
        Compiled Keras model.
    """
    dropout_rate = hp.Float("dropout_rate", min_value=0.2, max_value=0.5, step=0.1)
    l2_reg = hp.Choice("l2_reg", values=[1e-3, 1e-4, 1e-5])
    dense_units_1 = hp.Choice("dense_units_1", values=[128, 256, 512])
    dense_units_2 = hp.Choice("dense_units_2", values=[64, 128, 256])
    learning_rate = hp.Choice("learning_rate", values=[1e-3, 5e-4, 1e-4])

    model, base_model = build_model(
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        dense_units_1=dense_units_1,
        dense_units_2=dense_units_2,
    )
    freeze_backbone(base_model)

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
