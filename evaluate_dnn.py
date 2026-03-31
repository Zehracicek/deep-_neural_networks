"""
Train the NSL-KDD binary DNN (if not reusing saved weights) and evaluate on KDDTest.

Metrics: accuracy, precision, recall, F1, confusion matrix (attack = positive class).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from dnn_model import train_model
from preprocess_nsl_kdd import get_preprocessed_train_test


def predict_binary_labels(model, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Sigmoid output -> 0/1 predictions (1 = attack)."""
    proba = model.predict(X, batch_size=256, verbose=0).ravel()
    return (proba >= threshold).astype(np.int32)


def compute_test_metrics(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float | np.ndarray]:
    """Accuracy, precision, recall, F1, and confusion matrix on the test set (attack = positive)."""
    y_true = np.asarray(y_test).astype(np.int32).ravel()
    y_pred = predict_binary_labels(model, X_test, threshold=threshold)
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def evaluate_on_test(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    threshold: float = 0.5,
    confusion_matrix_path: Path | None = None,
) -> dict[str, float | np.ndarray]:
    results = compute_test_metrics(model, X_test, y_test, threshold=threshold)
    cm = results["confusion_matrix"]

    out_path = confusion_matrix_path or Path(__file__).resolve().parent / "confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal (0)", "Attack (1)"],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Test set confusion matrix")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix plot: {out_path}")

    return results


def print_metrics_table(results: dict[str, float | np.ndarray]) -> None:
    cm = results["confusion_matrix"]
    tn, fp, fn, tp = cm.ravel()
    print()
    print("=== Test set evaluation (KDDTest) ===")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}  (attacks predicted as attacks)")
    print(f"Recall:    {results['recall']:.4f}  (attacks caught)")
    print(f"F1-score:  {results['f1']:.4f}")
    print()
    print("Confusion matrix [rows = true, cols = predicted]")
    print("                 Pred Normal   Pred Attack")
    print(f"True Normal      {tn:>11}   {fp:>11}")
    print(f"True Attack      {fn:>11}   {tp:>11}")
    print()


def print_why_recall_matters_ids() -> None:
    print("=== Why recall matters in intrusion detection ===")
    print(
        "Recall (for the attack class) is the fraction of real attacks the model actually "
        "flags. A missed attack (false negative) means malicious traffic may run undetected, "
        "so the harm is direct. A false alarm (false positive) is costly too, but many IDS "
        "deployments prioritize catching intrusions over minimizing alerts, and tune thresholds "
        "to trade precision vs recall. High accuracy alone can hide poor attack recall when "
        "normals dominate the traffic mix, so recall (and F1) are standard IDS quality checks "
        "alongside the confusion matrix."
    )
    print()


def main() -> None:
    X_train, y_train, X_test, y_test = get_preprocessed_train_test()

    print("Training model (same setup as dnn_model: val_split=0.2, EarlyStopping)...")
    model, _history = train_model(
        X_train,
        y_train,
        epochs=20,
        validation_split=0.2,
        verbose=0,
        early_stopping_verbose=0,
    )

    results = evaluate_on_test(model, X_test, y_test)
    print_metrics_table(results)
    print_why_recall_matters_ids()


if __name__ == "__main__":
    main()
