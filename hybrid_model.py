"""
Hybrid LSTM + DNN multiclass model (Keras Functional API).

- DNN branch: current packet static features
- LSTM branch: sliding window of last N packets
- Output: Dense(5, softmax), sparse_categorical_crossentropy
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import callbacks, layers, optimizers

from preprocess_nsl_kdd import NUM_CLASSES
from sequence_utils import DEFAULT_WINDOW_SIZE


def multiclass_class_weights(y: np.ndarray) -> dict[int, float]:
    """Balanced class weights for Keras fit."""
    y_flat = np.asarray(y).astype(int).ravel()
    classes = np.arange(NUM_CLASSES)
    weights = compute_class_weight("balanced", classes=classes, y=y_flat)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def build_hybrid_model(
    input_dim: int,
    window_size: int = DEFAULT_WINDOW_SIZE,
    *,
    dnn_units: tuple[int, int, int] = (128, 64, 32),
    lstm_units: tuple[int, int] = (64, 32),
    dropout: float = 0.25,
    learning_rate: float = 0.001,
) -> keras.Model:
    """Functional API: static DNN + sequence LSTM -> softmax(5)."""
    static_in = layers.Input(shape=(input_dim,), name="static_features")
    seq_in = layers.Input(shape=(window_size, input_dim), name="packet_sequence")

    # --- DNN branch (static features) ---
    x = static_in
    for i, units in enumerate(dnn_units):
        x = layers.Dense(units, activation="relu", name=f"dnn_dense_{i + 1}")(x)
        x = layers.Dropout(dropout, name=f"dnn_dropout_{i + 1}")(x)
    dnn_out = layers.Dense(32, activation="relu", name="dnn_embedding")(x)

    # --- LSTM branch (temporal window) ---
    y = layers.LSTM(lstm_units[0], return_sequences=True, name="lstm_1")(seq_in)
    y = layers.Dropout(dropout, name="lstm_dropout_1")(y)
    y = layers.LSTM(lstm_units[1], name="lstm_2")(y)
    lstm_out = layers.Dense(32, activation="relu", name="lstm_embedding")(y)

    # --- Merge & classification head ---
    merged = layers.Concatenate(name="merge")([dnn_out, lstm_out])
    z = layers.Dense(64, activation="relu", name="head_dense")(merged)
    z = layers.Dropout(dropout, name="head_dropout")(z)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="class_probs")(z)

    model = keras.Model(inputs=[static_in, seq_in], outputs=outputs, name="hybrid_lstm_dnn")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_hybrid_model(
    X_train: np.ndarray,
    X_seq_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epochs: int = 20,
    batch_size: int = 256,
    validation_split: float = 0.2,
    validation_data: tuple | None = None,
    early_stopping_patience: int = 5,
    verbose: int | str = 1,
    class_weight: dict[int, float] | None = None,
    window_size: int = DEFAULT_WINDOW_SIZE,
    learning_rate: float = 0.001,
    dnn_units: tuple[int, int, int] = (128, 64, 32),
    lstm_units: tuple[int, int] = (64, 32),
    dropout: float = 0.25,
) -> tuple[keras.Model, callbacks.History]:
    """Train hybrid model; uses balanced class weights by default."""
    if class_weight is None:
        class_weight = multiclass_class_weights(y_train)

    model = build_hybrid_model(
        input_dim=X_train.shape[1],
        window_size=window_size,
        dnn_units=dnn_units,
        lstm_units=lstm_units,
        dropout=dropout,
        learning_rate=learning_rate,
    )
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=0,
    )
    fit_kw: dict[str, Any] = {
        "epochs": epochs,
        "batch_size": batch_size,
        "callbacks": [early_stop],
        "verbose": verbose,
        "class_weight": class_weight,
    }
    train_inputs = [X_train, X_seq_train]
    if validation_data is not None:
        X_v, X_seq_v, y_v = validation_data
        fit_kw["validation_data"] = ([X_v, X_seq_v], y_v)
        history = model.fit(train_inputs, y_train, **fit_kw)
    else:
        fit_kw["validation_split"] = validation_split
        history = model.fit(train_inputs, y_train, **fit_kw)
    return model, history


def predict_proba(model: keras.Model, X_static: np.ndarray, X_seq: np.ndarray) -> np.ndarray:
    """Softmax probabilities (n_samples, num_classes)."""
    return model.predict([X_static, X_seq], batch_size=256, verbose=0)


def predict_classes(model: keras.Model, X_static: np.ndarray, X_seq: np.ndarray) -> np.ndarray:
    proba = predict_proba(model, X_static, X_seq)
    return np.argmax(proba, axis=1).astype(np.int32)


def plot_training_history(
    history: callbacks.History,
    save_path: str | Path | None = None,
) -> None:
    h = history.history
    epochs = range(1, len(h["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, h["loss"], label="Eğitim kaybı")
    axes[0].plot(epochs, h["val_loss"], label="Doğrulama kaybı")
    axes[0].set_title("Sparse categorical crossentropy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, h["accuracy"], label="Eğitim doğruluğu")
    axes[1].plot(epochs, h["val_accuracy"], label="Doğrulama doğruluğu")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    out = Path(save_path) if save_path else Path(__file__).resolve().parent / "training_history.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
