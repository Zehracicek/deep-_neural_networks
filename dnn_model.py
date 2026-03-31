"""
Binary classification DNN for NSL-KDD (TensorFlow / Keras).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import callbacks, layers, optimizers

from preprocess_nsl_kdd import get_preprocessed_train_test


def build_binary_dnn(
    input_dim: int,
    hidden_units: tuple[int, int, int] = (128, 64, 32),
    dropout_rates: tuple[float, float, float] | None = None,
    learning_rate: float = 0.001,
) -> keras.Model:
    """
    Feedforward network: input -> 3x (Dense + ReLU [+ Dropout]) -> sigmoid output.

    Parameters
    ----------
    input_dim
        Number of features after preprocessing (one-hot + scaled numerics).
    hidden_units
        Sizes of the three hidden Dense layers.
    dropout_rates
        If set, a Dropout layer is inserted after each hidden ReLU (same length as hidden_units).
    learning_rate
        Adam learning rate.
    """
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


def train_model(
    X_train,
    y_train,
    *,
    epochs: int = 20,
    validation_split: float = 0.2,
    validation_data: tuple | None = None,
    batch_size: int = 256,
    early_stopping_patience: int = 5,
    early_stopping_verbose: int = 1,
    verbose: int | str = 1,
    class_weight: dict[int, float] | None = None,
    hidden_units: tuple[int, int, int] = (128, 64, 32),
    dropout_rates: tuple[float, float, float] | None = None,
    learning_rate: float = 0.001,
) -> tuple[keras.Model, keras.callbacks.History]:
    """
    Fit on training data. Either ``validation_split`` (fraction of X_train) or
    ``validation_data=(X_val, y_val)`` must be used for EarlyStopping on val_loss.
    """
    model = build_binary_dnn(
        input_dim=X_train.shape[1],
        hidden_units=hidden_units,
        dropout_rates=dropout_rates,
        learning_rate=learning_rate,
    )
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=early_stopping_verbose,
    )
    fit_kw: dict = {
        "epochs": epochs,
        "batch_size": batch_size,
        "callbacks": [early_stop],
        "verbose": verbose,
        "class_weight": class_weight,
    }
    if validation_data is not None:
        fit_kw["validation_data"] = validation_data
        history = model.fit(X_train, y_train, **fit_kw)
    else:
        fit_kw["validation_split"] = validation_split
        history = model.fit(X_train, y_train, **fit_kw)
    return model, history


def plot_training_history(
    history: keras.callbacks.History,
    save_path: str | Path | None = None,
) -> None:
    """Plot training vs validation loss and accuracy."""
    h = history.history
    n = len(h["loss"])
    epochs = range(1, n + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, h["loss"], label="Training loss")
    axes[0].plot(epochs, h["val_loss"], label="Validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Binary crossentropy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, h["accuracy"], label="Training accuracy")
    axes[1].plot(epochs, h["val_accuracy"], label="Validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(save_path) if save_path else Path(__file__).resolve().parent / "training_history.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out}")


def summarize_training(history: keras.callbacks.History) -> None:
    """Short readout of best epoch and whether train/val diverged."""
    h = history.history
    best = min(range(len(h["val_loss"])), key=lambda i: h["val_loss"][i])
    best_ep = best + 1
    print()
    print("--- Results (brief) ---")
    print(
        f"Best validation loss at epoch {best_ep}: "
        f"{h['val_loss'][best]:.4f} "
        f"(val accuracy {h['val_accuracy'][best]:.4f})."
    )
    print(
        f"At that epoch - train loss {h['loss'][best]:.4f}, "
        f"train accuracy {h['accuracy'][best]:.4f}."
    )
    last = len(h["loss"]) - 1
    if h["loss"][last] < h["val_loss"][last] and h["accuracy"][last] > h["val_accuracy"][last]:
        print(
            "Training loss is lower and training accuracy higher than validation at the "
            "last epoch - typical mild overfitting on the 80% train slice; EarlyStopping "
            "restored weights from the best val_loss epoch."
        )
    elif abs(h["loss"][last] - h["val_loss"][last]) < 0.02:
        print("Train and validation curves stayed close - model capacity and split look reasonable.")
    print(
        "NSL-KDD test set still contains unseen attack categories vs train; "
        "holdout test evaluation (not the 20% val split) is the stronger generalization check."
    )


def main() -> None:
    X_train, y_train, _, _ = get_preprocessed_train_test()
    model = build_binary_dnn(input_dim=X_train.shape[1])
    model.summary()

    model, history = train_model(
        X_train,
        y_train,
        epochs=20,
        validation_split=0.2,
    )
    plot_training_history(history)
    summarize_training(history)


if __name__ == "__main__":
    main()
