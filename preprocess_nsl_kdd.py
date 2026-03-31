"""
Preprocess NSL-KDD Parquet data for deep learning: scaling, one-hot encoding, binary labels.

Requires: pandas, pyarrow, scikit-learn, numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from load_nsl_kdd import assign_nsl_kdd_column_names

# Symbolic fields to one-hot encode (NSL-KDD standard).
CATEGORICAL_FEATURES: tuple[str, ...] = ("protocol_type", "service", "flag")

# Columns that must never appear in X (targets / leakage).
TARGET_AND_META: frozenset[str] = frozenset({"class", "classnum", "label"})


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


def binary_labels(df: pd.DataFrame) -> NDArray[np.int32]:
    """
    Binary intrusion labels: normal -> 0, any attack -> 1.
    Matching is case-insensitive on the target string column.
    """
    col = target_column(df)
    is_normal = df[col].astype(str).str.lower().eq("normal")
    y = (~is_normal).astype(np.int32)
    return y.to_numpy()


def infer_num_cat_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Feature columns = all columns except class / classnum / label.
    Categorical = intersection with NSL-KDD symbolic fields present in df.
    """
    features = [c for c in df.columns if c not in TARGET_AND_META]
    categorical = [c for c in CATEGORICAL_FEATURES if c in features]
    numerical = [c for c in features if c not in categorical]
    return numerical, categorical


def build_column_transformer(
    numerical: list[str],
    categorical: list[str],
) -> ColumnTransformer:
    """
    Scale numeric columns; one-hot encode categoricals.
    Unknown categories at transform time become all zeros (ignore).
    """
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
    """
    Fit preprocessing on training data, then transform train and test.
    Output X is float32 dense; y is int32 {0, 1}.
    """

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
        X = self._ct.transform(df)
        return np.asarray(X, dtype=np.float32)

    def fit_transform(self, df_train: pd.DataFrame) -> NDArray[np.float32]:
        self.fit(df_train)
        return self.transform(df_train)

    def feature_names_out(self) -> np.ndarray:
        return self.column_transformer.get_feature_names_out()


@dataclass
class PreparedNSLKDD:
    """Train/test matrices and labels ready for numpy/torch/keras."""

    X_train: NDArray[np.float32]
    X_test: NDArray[np.float32]
    y_train: NDArray[np.int32]
    y_test: NDArray[np.int32]
    numerical_columns: list[str]
    categorical_columns: list[str]
    feature_names: np.ndarray
    preprocessor: NSLKDDPreprocessor


def prepare_nsl_kdd_for_dl() -> PreparedNSLKDD:
    """
    End-to-end: load Parquet, fit preprocessor on train, transform both splits, binary y.
    X uses float32; y uses int32 in {0, 1}.
    """
    df_train, df_test = load_raw_frames()
    pre = NSLKDDPreprocessor()
    X_train = pre.fit_transform(df_train)
    X_test = pre.transform(df_test)
    y_train = binary_labels(df_train)
    y_test = binary_labels(df_test)
    return PreparedNSLKDD(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        numerical_columns=pre.numerical_columns,
        categorical_columns=pre.categorical_columns,
        feature_names=pre.feature_names_out(),
        preprocessor=pre,
    )


def get_preprocessed_train_test() -> tuple[
    NDArray[np.float32],
    NDArray[np.int32],
    NDArray[np.float32],
    NDArray[np.int32],
]:
    """
    Build train and test arrays from KDDTrain.parquet and KDDTest.parquet.

    Training rows come only from the train file; test rows only from the test file.
    The same preprocessing pipeline is fit on the train DataFrame once
    (StandardScaler + OneHotEncoder), then applied via transform to train and test,
    so both X splits live in the same feature space.
    """
    data = prepare_nsl_kdd_for_dl()
    return data.X_train, data.y_train, data.X_test, data.y_test


def main() -> None:
    data = prepare_nsl_kdd_for_dl()
    print("Numerical columns:", data.numerical_columns)
    print("Categorical columns (one-hot):", data.categorical_columns)
    print("Output feature dim:", data.X_train.shape[1])
    print("X_train:", data.X_train.shape, data.X_train.dtype)
    print("X_test:", data.X_test.shape, data.X_test.dtype)
    print("y_train:", data.y_train.shape, "unique:", np.unique(data.y_train))
    print("y_test:", data.y_test.shape, "unique:", np.unique(data.y_test))
    print("y_train balance (0,1):", np.bincount(data.y_train))
    print("y_test balance (0,1):", np.bincount(data.y_test))
    print("First 3 feature names:", data.feature_names[:3].tolist(), "...")


if __name__ == "__main__":
    main()
