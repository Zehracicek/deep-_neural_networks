"""
Preprocess NSL-KDD Parquet data for deep learning.

Supports:
  - Feature scaling + one-hot encoding (unchanged pipeline)
  - Binary labels (legacy): normal=0, attack=1
  - Multiclass labels (5 classes): Normal, DoS, Probe, R2L, U2R
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from load_nsl_kdd import assign_nsl_kdd_column_names

# =============================================================================
# Constants
# =============================================================================
CATEGORICAL_FEATURES: tuple[str, ...] = ("protocol_type", "service", "flag")
TARGET_AND_META: frozenset[str] = frozenset({"class", "classnum", "label"})

CLASS_NAMES: tuple[str, ...] = ("Normal", "DoS", "Probe", "R2L", "U2R")
NUM_CLASSES: int = len(CLASS_NAMES)
CLASS_NAME_TO_ID: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}

# NSL-KDD attack name -> category (lowercase keys)
_ATTACK_DOS = frozenset({"back", "land", "neptune", "pod", "smurf", "teardrop"})
_ATTACK_PROBE = frozenset({"ipsweep", "nmap", "portsweep", "satan"})
_ATTACK_R2L = frozenset(
    {
        "ftp_write",
        "guess_passwd",
        "imap",
        "multihop",
        "phf",
        "spy",
        "warezclient",
        "warezmaster",
    }
)
_ATTACK_U2R = frozenset({"buffer_overflow", "loadmodule", "perl", "rootkit"})

# KDDTest-only attack names (not in KDDTrain) -> category
_EXTRA_ATTACK_CATEGORY: dict[str, str] = {
    "apache2": "DoS",
    "mailbomb": "DoS",
    "processtable": "DoS",
    "udpstorm": "DoS",
    "worm": "DoS",
    "mscan": "Probe",
    "saint": "Probe",
    "httptunnel": "R2L",
    "named": "R2L",
    "ps": "R2L",
    "sendmail": "R2L",
    "snmpgetattack": "R2L",
    "snmpguess": "R2L",
    "sqlattack": "R2L",
    "xlock": "R2L",
    "xsnoop": "R2L",
    "xterm": "R2L",
}


def _data_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


def load_raw_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test Parquet and fix missing column names if needed."""
    d = _data_dir()
    train = assign_nsl_kdd_column_names(pd.read_parquet(d / "KDDTrain.parquet"))
    test = assign_nsl_kdd_column_names(pd.read_parquet(d / "KDDTest.parquet"))
    return train, test


def target_column(df: pd.DataFrame) -> str:
    if "class" in df.columns:
        return "class"
    if "label" in df.columns:
        return "label"
    raise KeyError("Expected a 'class' or 'label' column for the target.")


def attack_name_to_category(attack_name: str) -> str:
    """Map raw NSL-KDD attack label to one of CLASS_NAMES."""
    name = str(attack_name).strip().lower()
    if name == "normal":
        return "Normal"
    if name in _ATTACK_DOS:
        return "DoS"
    if name in _ATTACK_PROBE:
        return "Probe"
    if name in _ATTACK_R2L:
        return "R2L"
    if name in _ATTACK_U2R:
        return "U2R"
    if name in _EXTRA_ATTACK_CATEGORY:
        return _EXTRA_ATTACK_CATEGORY[name]
    # Bilinmeyen saldırı adları (genelde test kümesi) -> uzaktan erişim varsayımı
    return "R2L"


def binary_labels(df: pd.DataFrame) -> NDArray[np.int32]:
    """Binary intrusion labels: normal -> 0, any attack -> 1."""
    col = target_column(df)
    is_normal = df[col].astype(str).str.lower().eq("normal")
    return (~is_normal).astype(np.int32).to_numpy()


def multiclass_labels(df: pd.DataFrame) -> NDArray[np.int32]:
    """Multiclass integer labels in {0..4} for CLASS_NAMES order."""
    col = target_column(df)
    categories = df[col].astype(str).map(attack_name_to_category)
    return categories.map(CLASS_NAME_TO_ID).astype(np.int32).to_numpy()


def class_distribution(y: np.ndarray, labels: tuple[str, ...] = CLASS_NAMES) -> dict[str, int]:
    """Count per class name."""
    y = np.asarray(y).astype(int).ravel()
    counts = np.bincount(y, minlength=len(labels))
    return {labels[i]: int(counts[i]) for i in range(len(labels))}


def infer_num_cat_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    features = [c for c in df.columns if c not in TARGET_AND_META]
    categorical = [c for c in CATEGORICAL_FEATURES if c in features]
    numerical = [c for c in features if c not in categorical]
    return numerical, categorical


def build_column_transformer(
    numerical: list[str],
    categorical: list[str],
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


class NSLKDDPreprocessor:
    """Fit preprocessing on training data, then transform train and test."""

    def __init__(self) -> None:
        self._ct: ColumnTransformer | None = None
        self._numerical: list[str] = []
        self._categorical: list[str] = []

    @property
    def numerical_columns(self) -> list[str]:
        return list(self._numerical)

    @property
    def categorical_columns(self) -> list[str]:
        return list(self._categorical)

    @property
    def column_transformer(self) -> ColumnTransformer:
        if self._ct is None:
            raise RuntimeError("Call fit() before accessing column_transformer.")
        return self._ct

    def fit(self, df_train: pd.DataFrame) -> NSLKDDPreprocessor:
        self._numerical, self._categorical = infer_num_cat_columns(df_train)
        self._ct = build_column_transformer(self._numerical, self._categorical)
        self._ct.fit(df_train)
        return self

    def transform(self, df: pd.DataFrame) -> NDArray[np.float32]:
        if self._ct is None:
            raise RuntimeError("Call fit() before transform().")
        return np.asarray(self._ct.transform(df), dtype=np.float32)

    def fit_transform(self, df_train: pd.DataFrame) -> NDArray[np.float32]:
        self.fit(df_train)
        return self.transform(df_train)

    def feature_names_out(self) -> np.ndarray:
        return self.column_transformer.get_feature_names_out()


@dataclass
class PreparedNSLKDD:
    """Train/test matrices — binary or multiclass labels."""

    X_train: NDArray[np.float32]
    X_test: NDArray[np.float32]
    y_train: NDArray[np.int32]
    y_test: NDArray[np.int32]
    numerical_columns: list[str]
    categorical_columns: list[str]
    feature_names: np.ndarray
    preprocessor: NSLKDDPreprocessor
    class_names: tuple[str, ...] = CLASS_NAMES
    multiclass: bool = False
    smote_before: dict[str, int] = field(default_factory=dict)
    smote_after: dict[str, int] = field(default_factory=dict)


def prepare_nsl_kdd_for_dl(*, multiclass: bool = True, apply_smote: bool = False) -> PreparedNSLKDD:
    """
    End-to-end preprocessing.

    Parameters
    ----------
    multiclass
        If True (default), y in {0..4} for Normal/DoS/Probe/R2L/U2R.
        If False, legacy binary labels.
    apply_smote
        If True, SMOTE is applied on training rows only (see smote_balance.py).
    """
    df_train, df_test = load_raw_frames()
    pre = NSLKDDPreprocessor()
    X_train = pre.fit_transform(df_train)
    X_test = pre.transform(df_test)

    if multiclass:
        y_train = multiclass_labels(df_train)
        y_test = multiclass_labels(df_test)
    else:
        y_train = binary_labels(df_train)
        y_test = binary_labels(df_test)

    smote_before: dict[str, int] = {}
    smote_after: dict[str, int] = {}

    if apply_smote and multiclass:
        from smote_balance import apply_smote_train

        smote_before = class_distribution(y_train)
        X_train, y_train = apply_smote_train(X_train, y_train)
        smote_after = class_distribution(y_train)

    return PreparedNSLKDD(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        numerical_columns=pre.numerical_columns,
        categorical_columns=pre.categorical_columns,
        feature_names=pre.feature_names_out(),
        preprocessor=pre,
        class_names=CLASS_NAMES if multiclass else ("Normal", "Attack"),
        multiclass=multiclass,
        smote_before=smote_before,
        smote_after=smote_after,
    )


@dataclass
class PreparedHybrid:
    """Static features + LSTM sequences for hybrid model."""

    X_train: NDArray[np.float32]
    X_seq_train: NDArray[np.float32]
    y_train: NDArray[np.int32]
    X_test: NDArray[np.float32]
    X_seq_test: NDArray[np.float32]
    y_test: NDArray[np.int32]
    feature_names: np.ndarray
    numerical_columns: list[str]
    categorical_columns: list[str]
    preprocessor: NSLKDDPreprocessor
    class_names: tuple[str, ...]
    window_size: int
    smote_before: dict[str, int] = field(default_factory=dict)
    smote_after: dict[str, int] = field(default_factory=dict)
    valid_train_indices: NDArray[np.int32] = field(default_factory=lambda: np.array([], dtype=np.int32))
    valid_test_indices: NDArray[np.int32] = field(default_factory=lambda: np.array([], dtype=np.int32))


def prepare_hybrid_dataset(
    *,
    window_size: int | None = None,
    apply_smote: bool = True,
) -> PreparedHybrid:
    """Full pipeline: preprocess -> optional SMOTE -> sliding windows."""
    from sequence_utils import DEFAULT_WINDOW_SIZE, build_sequence_tensors

    win = window_size or DEFAULT_WINDOW_SIZE
    base = prepare_nsl_kdd_for_dl(multiclass=True, apply_smote=apply_smote)

    X_tr, X_seq_tr, y_tr, valid_tr = build_sequence_tensors(base.X_train, base.y_train, win)
    X_te, X_seq_te, y_te, valid_te = build_sequence_tensors(base.X_test, base.y_test, win)

    return PreparedHybrid(
        X_train=X_tr,
        X_seq_train=X_seq_tr,
        y_train=y_tr,
        X_test=X_te,
        X_seq_test=X_seq_te,
        y_test=y_te,
        feature_names=base.feature_names,
        numerical_columns=base.numerical_columns,
        categorical_columns=base.categorical_columns,
        preprocessor=base.preprocessor,
        class_names=base.class_names,
        window_size=win,
        smote_before=base.smote_before,
        smote_after=base.smote_after,
        valid_train_indices=valid_tr,
        valid_test_indices=valid_te,
    )


def get_preprocessed_train_test(
    *,
    multiclass: bool = True,
    apply_smote: bool = False,
) -> tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.float32], NDArray[np.int32]]:
    """Backward-compatible tuple return for scripts that expect four arrays."""
    data = prepare_nsl_kdd_for_dl(multiclass=multiclass, apply_smote=apply_smote)
    return data.X_train, data.y_train, data.X_test, data.y_test


def main() -> None:
    data = prepare_nsl_kdd_for_dl(multiclass=True, apply_smote=False)
    print("Sınıflar:", data.class_names)
    print("Özellik boyutu:", data.X_train.shape[1])
    print("Eğitim dağılımı:", class_distribution(data.y_train))
    print("Test dağılımı:", class_distribution(data.y_test))


if __name__ == "__main__":
    main()