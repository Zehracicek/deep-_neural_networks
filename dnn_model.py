"""
Model training entry points.

- **Hybrid multiclass (default):** see ``hybrid_model.py`` (LSTM + DNN, 5 classes).
- **Legacy binary DNN:** ``build_binary_dnn`` / ``train_model`` kept for backward compatibility.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import callbacks, layers, optimizers

# Re-export hybrid API as primary path
from hybrid_model import (  # noqa: F401
    build_hybrid_model,
    multiclass_class_weights,
    predict_classes,
    predict_proba,
    train_hybrid_model,
)
from preprocess_nsl_kdd import get_preprocessed_train_test


def build_binary_dnn(
    input_dim: int,
    hidden_units: tuple[int, int, int] = (128, 64, 32),
    dropout_rates: tuple[float, float, float] | None = None,
    learning_rate: float = 0.001,
) -> keras.Model:
    """Legacy binary classifier (sigmoid). Prefer ``build_hybrid_model``."""
    blocks: list = [layers.Input(shape=(input_dim,), name="features")]
    for i, units in enumerate(hidden_units):
        blocks.append(layers.Dense(units, activation="relu", name=f"hidden_{i + 1}"))
        if dropout_rates is not None:
            blocks.append(layers.Dropout(dropout_rates[i], name=f"dropout_{i + 1}"))
    blocks.append(layers.Dense(1, activation="sigmoid", name="prob_attack"))
    model = keras.Sequential(blocks, name="nsl_kdd_binary_dnn")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(X_train, y_train, **kwargs) -> tuple[keras.Model, callbacks.History]:
    """Legacy binary training. For multiclass hybrid, use ``train_hybrid_model``."""
    model = build_binary_dnn(
        input_dim=X_train.shape[1],
        hidden_units=kwargs.pop("hidden_units", (128, 64, 32)),
        dropout_rates=kwargs.pop("dropout_rates", None),
        learning_rate=kwargs.pop("learning_rate", 0.001),
    )
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=kwargs.pop("early_stopping_patience", 5),
        restore_best_weights=True,
        verbose=kwargs.pop("early_stopping_verbose", 1),
    )
    fit_kw = {
        "epochs": kwargs.pop("epochs", 20),
        "batch_size": kwargs.pop("batch_size", 256),
        "callbacks": [early_stop],
        "verbose": kwargs.pop("verbose", 1),
        "class_weight": kwargs.pop("class_weight", None),
    }
    validation_data = kwargs.pop("validation_data", None)
    if validation_data is not None:
        fit_kw["validation_data"] = validation_data
        history = model.fit(X_train, y_train, **fit_kw)
    else:
        fit_kw["validation_split"] = kwargs.pop("validation_split", 0.2)
        history = model.fit(X_train, y_train, **fit_kw)
    return model, history


def main() -> None:
    from preprocess_nsl_kdd import prepare_hybrid_dataset

    data = prepare_hybrid_dataset(apply_smote=True)
    model, history = train_hybrid_model(
        data.X_train,
        data.X_seq_train,
        data.y_train,
        epochs=10,
        validation_split=0.2,
        window_size=data.window_size,
    )
    model.summary()
    h = history.history
    fig, ax = plt.subplots()
    ax.plot(h["loss"], label="loss")
    ax.plot(h["val_loss"], label="val_loss")
    ax.legend()
    fig.savefig(Path(__file__).parent / "training_history.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
