"""
Exploratory data analysis for NSL-KDD Parquet train/test splits.

Run: python eda_nsl_kdd.py
"""

from pathlib import Path

import pandas as pd

from load_nsl_kdd import assign_nsl_kdd_column_names

# NSL-KDD symbolic fields (string-like in raw data); used if dtypes are numeric by mistake.
NSL_KDD_SYMBOLIC = {"protocol_type", "service", "flag"}


def label_column_name(df: pd.DataFrame) -> str:
    """Target column: `label` in full NSL-KDD CSV; this repo uses `class`."""
    if "label" in df.columns:
        return "label"
    if "class" in df.columns:
        return "class"
    raise KeyError("No label column found (expected 'label' or 'class').")


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(__file__).resolve().parent / "data"
    train = pd.read_parquet(data_dir / "KDDTrain.parquet")
    test = pd.read_parquet(data_dir / "KDDTest.parquet")
    return assign_nsl_kdd_column_names(train), assign_nsl_kdd_column_names(test)


def feature_columns(df: pd.DataFrame, target: str) -> list[str]:
    """Columns used as inputs: drop target and numeric class id if present."""
    drop = {target}
    if "classnum" in df.columns:
        drop.add("classnum")
    return [c for c in df.columns if c not in drop]


def infer_categorical_numerical(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    """
    Categorical: object/category/string dtypes, or known symbolic NSL fields.
    Numerical: remaining feature columns (includes binary 0/1 counts as numeric).
    """
    categorical: list[str] = []
    numerical: list[str] = []
    for col in features:
        s = df[col]
        if col in NSL_KDD_SYMBOLIC:
            categorical.append(col)
            continue
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            categorical.append(col)
        elif isinstance(s.dtype, pd.CategoricalDtype):
            categorical.append(col)
        else:
            numerical.append(col)
    categorical = sorted(set(categorical), key=features.index)
    numerical = sorted(set(numerical), key=features.index)
    return categorical, numerical


def print_class_distribution(df: pd.DataFrame, name: str, target: str) -> None:
    counts = df[target].value_counts()
    pct = df[target].value_counts(normalize=True).mul(100).round(2)
    dist = pd.DataFrame({"count": counts, "percent": pct})
    print(f"=== Class distribution ({name}) - column `{target}` ===")
    print(dist.to_string())
    print(f"Total rows: {len(df)}")
    print()


def print_missing(df: pd.DataFrame, name: str) -> None:
    missing = df.isna().sum()
    missing = missing[missing > 0]
    print(f"=== Missing values ({name}) ===")
    if missing.empty:
        print("No missing values.")
    else:
        print(missing.to_string())
    print(f"Total NA cells: {df.isna().sum().sum()}")
    print()


def main() -> None:
    df_train, df_test = load_frames()
    target = label_column_name(df_train)
    feats_train = feature_columns(df_train, target)
    categorical, numerical = infer_categorical_numerical(df_train, feats_train)

    print("=== EDA: NSL-KDD (Parquet) ===\n")

    # Class distribution (label / class)
    print_class_distribution(df_train, "KDDTrain", target)
    print_class_distribution(df_test, "KDDTest", target)

    # Categorical vs numerical features
    print("=== Feature types (train, excluding target / classnum) ===")
    print(f"Categorical ({len(categorical)}): {categorical}")
    print(f"Numerical ({len(numerical)}): {numerical}")
    print()

    # Missing values
    print_missing(df_train, "KDDTrain")
    print_missing(df_test, "KDDTest")

    # Basic statistics: numeric features + quick categorical summaries
    print("=== Basic statistics - numerical features (KDDTrain) ===")
    if numerical:
        print(df_train[numerical].describe().T.to_string())
    else:
        print("(none)")
    print()

    print("=== Categorical features - unique counts (KDDTrain) ===")
    for col in categorical:
        nuniq = df_train[col].nunique(dropna=False)
        print(f"  {col}: {nuniq} unique values")
    print()

    print("=== Sample value_counts (top 5 per categorical, KDDTrain) ===")
    for col in categorical:
        print(f"--- {col} ---")
        print(df_train[col].value_counts().head(5).to_string())
        print()

    print(
        """=== Preprocessing typically needed for this dataset ===

1. Target definition
   - Use string `label` or `class` for multi-class attack names, or `classnum` for
     integer labels. For binary intrusion detection, map all attacks to "attack" and
     keep "normal" as negative class (train and test separately for consistent rules).

2. Categorical encoding
   - Encode `protocol_type`, `service`, and `flag` (high cardinality on `service`).
   - Options: one-hot encoding, target encoding, or embeddings. Fit encoders on TRAIN
     only and apply to TEST to avoid leakage.

3. Numeric scaling
   - Features span very different scales (e.g. `duration`, `src_bytes` vs rates in [0,1]).
   - StandardScaler or MinMaxScaler (fit on train) usually helps gradient-based models.

4. Redundant target column
   - If both `class` and `classnum` exist, drop one from features so the model does not
     see the answer; keep one as the supervised target.

5. Train vs test distribution
   - NSL-KDD test contains attack types not seen in train; treat generalization and
     evaluation (e.g. per-class metrics) accordingly.

6. Missing values
   - If none are found, no imputation is required; if they appear in other exports,
     use median/mode fit on train.

7. Optional
   - Remove exact duplicate rows if desired; check for constant or near-constant features.
"""
    )


if __name__ == "__main__":
    main()
