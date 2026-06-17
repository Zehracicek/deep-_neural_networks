"""
SMOTE oversampling for imbalanced multiclass training data.
"""

from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE
from numpy.typing import NDArray

from preprocess_nsl_kdd import NUM_CLASSES, class_distribution


def _smote_k_neighbors(y: np.ndarray) -> int:
    """Pick k_neighbors so SMOTE works with very small minority classes (e.g. U2R)."""
    counts = np.bincount(np.asarray(y).astype(int).ravel(), minlength=NUM_CLASSES)
    minority = int(counts[counts > 0].min())
    return max(1, min(5, minority - 1))


def apply_smote_train(
    X_train: NDArray[np.float32],
    y_train: NDArray[np.int32],
    *,
    random_state: int = 42,
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Apply SMOTE on training features only; returns resampled (X, y)."""
    k = _smote_k_neighbors(y_train)
    smote = SMOTE(random_state=random_state, k_neighbors=k)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return np.asarray(X_res, dtype=np.float32), np.asarray(y_res, dtype=np.int32)


def print_smote_report(before: dict[str, int], after: dict[str, int]) -> None:
    print("=== SMOTE sınıf dağılımı ===")
    print(f"{'Sınıf':<10} {'Önce':>10} {'Sonra':>10}")
    for name in before:
        print(f"{name:<10} {before[name]:>10,} {after.get(name, 0):>10,}")
    print()
