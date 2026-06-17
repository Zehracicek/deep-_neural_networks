"""
Evaluate hybrid multiclass IDS on KDDTest.

Metrics: F1 (macro), ROC-AUC (OvR), per-class precision/recall, confusion matrix.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from hybrid_model import predict_classes, predict_proba, train_hybrid_model
from preprocess_nsl_kdd import CLASS_NAMES, NUM_CLASSES, class_distribution, prepare_hybrid_dataset
from smote_balance import print_smote_report


def compute_multiclass_metrics(
    model,
    X_static: np.ndarray,
    X_seq: np.ndarray,
    y_true: np.ndarray,
    *,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> dict:
    """Full multiclass metric bundle."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_proba = predict_proba(model, X_static, X_seq)
    y_pred = np.argmax(y_proba, axis=1).astype(int)

    labels = list(range(len(class_names)))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec, rec, f1_per, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    y_bin = label_binarize(y_true, classes=labels)
    try:
        roc_auc = float(
            roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
        )
    except ValueError:
        roc_auc = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true, y_pred, target_names=list(class_names), zero_division=0
    )

    return {
        "f1_macro": f1_macro,
        "roc_auc_ovr_macro": roc_auc,
        "precision_per_class": prec,
        "recall_per_class": rec,
        "f1_per_class": f1_per,
        "support_per_class": support,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "classification_report": report,
        "class_names": class_names,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: tuple[str, ...],
    save_path: Path | None = None,
) -> None:
    out = save_path or Path(__file__).resolve().parent / "confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_names))
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
    ax.set_title("Test kümesi — sınıf başına karışıklık matrisi")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Kaydedildi: {out}")


def print_metrics_summary(results: dict) -> None:
    names = results["class_names"]
    print()
    print("=== Test kümesi değerlendirmesi (çok sınıflı) ===")
    print(f"F1-score (macro):     {results['f1_macro']:.4f}")
    print(f"ROC-AUC (OvR, macro): {results['roc_auc_ovr_macro']:.4f}")
    print()
    print(f"{'Sınıf':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    for i, name in enumerate(names):
        print(
            f"{name:<10} {results['precision_per_class'][i]:10.4f} "
            f"{results['recall_per_class'][i]:10.4f} "
            f"{results['f1_per_class'][i]:10.4f} "
            f"{int(results['support_per_class'][i]):10,}"
        )
    print()
    print(results["classification_report"])


# Legacy binary API (deprecated — kept for old scripts)
def predict_binary_labels(model, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Deprecated: binary sigmoid models only."""
    proba = model.predict(X, batch_size=256, verbose=0).ravel()
    return (proba >= threshold).astype(np.int32)


def compute_test_metrics(model, X_test, y_test, *, threshold: float = 0.5) -> dict:
    """Deprecated: use compute_multiclass_metrics with hybrid inputs."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    y_pred = predict_binary_labels(model, X_test, threshold)
    y_true = np.asarray(y_test).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def evaluate_on_test(model, X_test, y_test, **kwargs) -> dict:
    """Deprecated wrapper."""
    return compute_test_metrics(model, X_test, y_test, **kwargs)


def main() -> None:
    print("Hibrit veri hattı yükleniyor (SMOTE + diziler)…")
    data = prepare_hybrid_dataset(apply_smote=True)
    if data.smote_before:
        print_smote_report(data.smote_before, data.smote_after)

    print("Eğitim dağılımı:", class_distribution(data.y_train))
    print("Model eğitiliyor…")
    model, _ = train_hybrid_model(
        data.X_train,
        data.X_seq_train,
        data.y_train,
        epochs=15,
        validation_split=0.2,
        verbose=1,
        window_size=data.window_size,
    )

    results = compute_multiclass_metrics(
        model, data.X_test, data.X_seq_test, data.y_test, class_names=data.class_names
    )
    print_metrics_summary(results)
    plot_confusion_matrix(results["confusion_matrix"], data.class_names)


if __name__ == "__main__":
    main()
