"""
Load NSL-KDD train/test splits from Parquet and inspect basic structure.

Requires: pandas, pyarrow (or fastparquet).
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Step 0: Canonical NSL-KDD column names (full KDD Cup / NSL-KDD layout).
# The original CSV has 41 continuous/symbolic features plus one label column.
# We use this when Parquet was saved without column metadata (e.g. default 0..N-1).
# ---------------------------------------------------------------------------
NSL_KDD_COLUMNS_42 = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
]

# This repo's Parquet files use a 38-column subset (some features omitted) plus
# `class` (name) and `classnum` (numeric). Use only when names are missing and n==38.
NSL_KDD_COLUMNS_38_PARQUET = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "serror_rate",
    "srv_serror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "dst_host_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "class",
    "classnum",
]


def column_names_missing(df: pd.DataFrame) -> bool:
    """
    Detect whether the frame has no real column names (e.g. Parquet/CSV export
    without headers). We treat integer 0..n-1, all-'Unnamed', or all-blank as missing.
    """
    cols = list(df.columns)
    if not cols:
        return True
    if all(str(c).strip() == "" for c in cols):
        return True
    if all(str(c).lower().startswith("unnamed") for c in cols):
        return True
    # Default RangeIndex-style integer names 0, 1, 2, ...
    try:
        if all(int(c) == i for i, c in enumerate(cols)):
            return True
    except (TypeError, ValueError):
        pass
    return False


def assign_nsl_kdd_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    If names are missing, assign NSL-KDD names matching the column count:
    42 -> full Cup layout; 38 -> this project's subset + class/classnum.
    """
    if not column_names_missing(df):
        return df
    n = df.shape[1]
    if n == len(NSL_KDD_COLUMNS_42):
        df = df.copy()
        df.columns = NSL_KDD_COLUMNS_42
    elif n == len(NSL_KDD_COLUMNS_38_PARQUET):
        df = df.copy()
        df.columns = NSL_KDD_COLUMNS_38_PARQUET
    else:
        raise ValueError(
            f"Column names missing but got {n} columns; expected 42 (full NSL-KDD) "
            f"or 38 (this repo's Parquet layout). Update name lists in load_nsl_kdd.py."
        )
    return df


def main() -> None:
    # Resolve paths relative to this script so the project can be run from any cwd.
    data_dir = Path(__file__).resolve().parent / "data"
    train_path = data_dir / "KDDTrain.parquet"
    test_path = data_dir / "KDDTest.parquet"

    # -----------------------------------------------------------------------
    # Step 1: Read Parquet files with pandas.
    # read_parquet uses the pyarrow engine by default when installed; it preserves
    # dtypes and column names stored in the file metadata.
    # -----------------------------------------------------------------------
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    # -----------------------------------------------------------------------
    # Step 2: If column names are missing, assign proper NSL-KDD names.
    # -----------------------------------------------------------------------
    df_train = assign_nsl_kdd_column_names(df_train)
    df_test = assign_nsl_kdd_column_names(df_test)

    # -----------------------------------------------------------------------
    # Step 3: Display the first 5 rows of each split (quick sanity check).
    # -----------------------------------------------------------------------
    print("=== KDDTrain - first 5 rows ===")
    print(df_train.head(5))
    print()

    print("=== KDDTest - first 5 rows ===")
    print(df_test.head(5))
    print()

    # -----------------------------------------------------------------------
    # Step 4: Show dataset shape (rows, columns) for train and test.
    # -----------------------------------------------------------------------
    print("=== Dataset shapes ===")
    print(f"KDDTrain: {df_train.shape[0]} rows x {df_train.shape[1]} columns")
    print(f"KDDTest:  {df_test.shape[0]} rows x {df_test.shape[1]} columns")
    print()

    # -----------------------------------------------------------------------
    # Step 5: Show column names.
    # -----------------------------------------------------------------------
    print("=== Column names (KDDTrain) ===")
    print(list(df_train.columns))
    print()
    print("=== Column names (KDDTest) ===")
    print(list(df_test.columns))


if __name__ == "__main__":
    main()
