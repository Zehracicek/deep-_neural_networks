"""
Compare baseline NSL-KDD DNN vs improved setup: class weights, dropout, tuned hyperparameters.

Baseline: no dropout, no class_weight (original recipe).
Improved: grid search with a fixed stratified validation split; best config chosen by
validation attack recall (IDS-oriented), tie-broken by lower validation loss.
Final improved model is retrained with the same hyperparameters + class_weight.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import utils as keras_utils

from dnn_model import train_model
from evaluate_dnn import compute_test_metrics
from preprocess_nsl_kdd import get_preprocessed_train_test


def binary_class_weights(y: np.ndarray) -> dict[int, float]:
    """Keras `class_weight` dict for labels 0/1 (sklearn balanced formula)."""
    y_flat = np.asarray(y).astype(int).ravel()
    classes = np.array([0, 1])
    w = compute_class_weight("balanced", classes=classes, y=y_flat)
    return {int(c): float(wi) for c, wi in zip(classes, w)}


@dataclass(frozen=True)
class TuneConfig:
    label: str
    dropout_rates: tuple[float, float, float]
    learning_rate: float
    hidden_units: tuple[int, int, int]


def best_val_loss(history) -> float:
    return float(min(history.history["val_loss"]))


def main() -> None:
    X_train, y_train, X_test, y_test = get_preprocessed_train_test()
    cw = binary_class_weights(y_train)

    # Fixed stratified train/val so every tuning trial sees the same validation fold.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42,
    )

    print("Class counts (full train): normal(0) =", int((y_train == 0).sum()), " attack(1) =", int((y_train == 1).sum()))
    print("Balanced class_weight:", cw)
    print(f"Inner split for tuning: fit on {len(X_tr):,} rows, val on {len(X_val):,} rows (stratified).")
    print()

    # --- Baseline (before): no dropout, no class weighting ---
    print("=== Baseline (before improvements) ===")
    keras_utils.set_random_seed(42)
    base_model, base_hist = train_model(
        X_tr,
        y_tr,
        epochs=20,
        validation_data=(X_val, y_val),
        verbose=0,
        early_stopping_verbose=0,
        class_weight=None,
        dropout_rates=None,
        hidden_units=(128, 64, 32),
        learning_rate=0.001,
    )
    base_metrics = compute_test_metrics(base_model, X_test, y_test)
    base_val = compute_test_metrics(base_model, X_val, y_val)
    print(
        f"Test (baseline)   acc={base_metrics['accuracy']:.4f}  prec={base_metrics['precision']:.4f}  "
        f"rec={base_metrics['recall']:.4f}  f1={base_metrics['f1']:.4f}  "
        f"| val rec={base_val['recall']:.4f}  best val_loss={best_val_loss(base_hist):.4f}"
    )
    print()

    candidates: list[TuneConfig] = [
        TuneConfig("d0.25_lr1e3_h128", (0.25, 0.25, 0.25), 1e-3, (128, 64, 32)),
        TuneConfig("d0.35_lr5e4_h128", (0.35, 0.35, 0.35), 5e-4, (128, 64, 32)),
        TuneConfig("d0.2_lr1e3_h256", (0.2, 0.2, 0.2), 1e-3, (256, 128, 64)),
        TuneConfig("d0.3_lr1e3_h64", (0.3, 0.3, 0.3), 1e-3, (64, 64, 32)),
        TuneConfig("d0.15_lr1e3_h128", (0.15, 0.15, 0.15), 1e-3, (128, 64, 32)),
    ]

    print("=== Hyperparameter search (metric: maximize val attack recall; tie-break: lower val_loss) ===")
    best_cfg: TuneConfig | None = None
    best_score: tuple[float, float] = (-1.0, float("inf"))  # (val_recall, val_loss); tie-break via tuple compare

    for i, cfg in enumerate(candidates):
        keras_utils.set_random_seed(100 + i)
        model, hist = train_model(
            X_tr,
            y_tr,
            epochs=20,
            validation_data=(X_val, y_val),
            verbose=0,
            early_stopping_verbose=0,
            class_weight=cw,
            dropout_rates=cfg.dropout_rates,
            hidden_units=cfg.hidden_units,
            learning_rate=cfg.learning_rate,
        )
        vm = compute_test_metrics(model, X_val, y_val)
        tm = compute_test_metrics(model, X_test, y_test)
        vloss = best_val_loss(hist)
        score = (vm["recall"], -vloss)  # max recall, then max -vloss
        print(
            f"  [{cfg.label}] val_rec={vm['recall']:.4f}  val_loss={vloss:.4f}  "
            f"test_rec={tm['recall']:.4f}  test_f1={tm['f1']:.4f}"
        )
        if score > (best_score[0], -best_score[1]):
            best_score = (vm["recall"], vloss)
            best_cfg = cfg

    assert best_cfg is not None
    print(
        f"\nChosen: {best_cfg.label}  (val recall={best_score[0]:.4f}, "
        f"best val_loss during that run={best_score[1]:.4f})"
    )
    print()

    # --- Improved (after): same inner split + full training artifact for test comparison ---
    print("=== Improved (after): best hparams + class_weight + dropout (same 80% fit / 20% val) ===")
    keras_utils.set_random_seed(999)
    improved_model, improved_hist = train_model(
        X_tr,
        y_tr,
        epochs=20,
        validation_data=(X_val, y_val),
        verbose=0,
        early_stopping_verbose=0,
        class_weight=cw,
        dropout_rates=best_cfg.dropout_rates,
        hidden_units=best_cfg.hidden_units,
        learning_rate=best_cfg.learning_rate,
    )
    imp_metrics = compute_test_metrics(improved_model, X_test, y_test)
    print(
        f"Test (improved)   acc={imp_metrics['accuracy']:.4f}  prec={imp_metrics['precision']:.4f}  "
        f"rec={imp_metrics['recall']:.4f}  f1={imp_metrics['f1']:.4f}  "
        f"| best val_loss={best_val_loss(improved_hist):.4f}"
    )
    print(f"  dropout={best_cfg.dropout_rates}  lr={best_cfg.learning_rate}  hidden={best_cfg.hidden_units}")
    print()

    print("=== Comparison on KDDTest (threshold 0.5) ===")
    print(f"{'':22}  {'Acc':>8}  {'Prec':>8}  {'Rec':>8}  {'F1':>8}")
    print(
        f"{'Baseline':<22}  {base_metrics['accuracy']:8.4f}  {base_metrics['precision']:8.4f}  "
        f"{base_metrics['recall']:8.4f}  {base_metrics['f1']:8.4f}"
    )
    print(
        f"{'Improved':<22}  {imp_metrics['accuracy']:8.4f}  {imp_metrics['precision']:8.4f}  "
        f"{imp_metrics['recall']:8.4f}  {imp_metrics['f1']:8.4f}"
    )
    print()
    print(
        f"Delta (improved - baseline): recall {imp_metrics['recall'] - base_metrics['recall']:+.4f}, "
        f"F1 {imp_metrics['f1'] - base_metrics['f1']:+.4f}, "
        f"accuracy {imp_metrics['accuracy'] - base_metrics['accuracy']:+.4f}"
    )
    print(
        "Tuning targeted validation attack recall with balanced class_weight and dropout; "
        "expect higher recall with some precision/accuracy trade-off vs an unweighted baseline."
    )


if __name__ == "__main__":
    main()
