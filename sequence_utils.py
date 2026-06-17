"""
Sliding-window sequence builder for the LSTM branch.

Each sample uses the last ``window_size`` packets (rows) as a sequence;
the label is taken from the final row in the window.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

DEFAULT_WINDOW_SIZE = 10


def build_sequence_tensors(
    X: NDArray[np.float32],
    y: NDArray[np.int32],
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int32], NDArray[np.int32]]:
    """
    Build aligned static + sequence tensors.

    Returns
    -------
    X_static : (n_samples, n_features)
    X_seq : (n_samples, window_size, n_features)
    y_out : (n_samples,)
    valid_indices : original row indices in X for each sample (end of window)
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32).ravel()
    n, f = X.shape
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if n < window_size:
        raise ValueError(f"Need at least {window_size} rows, got {n}")

    n_samples = n - window_size + 1
    X_seq = np.zeros((n_samples, window_size, f), dtype=np.float32)
    valid_indices = np.arange(window_size - 1, n, dtype=np.int32)

    for i in range(n_samples):
        X_seq[i] = X[i : i + window_size]

    X_static = X[valid_indices]
    y_out = y[valid_indices]
    return X_static, X_seq, y_out, valid_indices


def sequence_for_index(
    X: NDArray[np.float32],
    index: int,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Single static row + sequence ending at ``index`` (pad start if needed)."""
    X = np.asarray(X, dtype=np.float32)
    index = int(index)
    if index >= window_size - 1:
        seq = X[index - window_size + 1 : index + 1]
    else:
        pad = window_size - index - 1
        seq = np.vstack([np.tile(X[0], (pad, 1)), X[: index + 1]])
    return X[index : index + 1], seq[np.newaxis, ...].astype(np.float32)
