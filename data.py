
import pandas as pd

from .config import (
    EXCLUDED_COLUMNS,
    TRIP_COL,
    OBS_COL,
    ARC_COL,
    TARGET_COL,
)


def load_candidate_data(path):
    df = pd.read_csv(path)

    required = {
        TRIP_COL,
        OBS_COL,
        ARC_COL,
        TARGET_COL,
        "dist_m",
        "heading_cos",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df.isna().any().any():
        raise ValueError("Dataset contains missing values.")

    if not set(df[TARGET_COL].unique()).issubset({0, 1}):
        raise ValueError("Labels must be binary: 0 or 1.")

    feature_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in EXCLUDED_COLUMNS
    ]

    if not feature_cols:
        raise ValueError("No usable numeric features found.")

    return df, feature_cols
