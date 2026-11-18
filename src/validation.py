
import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import TRIP_COL, OBS_COL, ARC_COL, TARGET_COL, N_SPLITS


def make_grouped_folds(X, y, groups, n_splits=N_SPLITS):
    """
    Create GroupKFold splits using trip_id.
    Ensures that no trip appears in both training and validation.
    """

    groups = pd.Series(groups).reset_index(drop=True)

    if len(X) != len(y) or len(X) != len(groups):
        raise ValueError("X, y and groups must have the same number of rows.")

    unique_groups = groups.nunique()

    if unique_groups < n_splits:
        raise ValueError(
            f"Need at least {n_splits} unique trips; found {unique_groups}."
        )

    splitter = GroupKFold(n_splits=n_splits)

    folds = list(splitter.split(X, y, groups))

    for fold_no, (train_idx, val_idx) in enumerate(folds, start=1):

        train_groups = set(groups.iloc[train_idx])
        val_groups = set(groups.iloc[val_idx])

        overlap = train_groups.intersection(val_groups)

        if overlap:
            raise RuntimeError(
                f"Trip leakage detected in fold {fold_no}: {overlap}"
            )

        print(
            f"Fold {fold_no}: "
            f"train={len(train_idx)} rows/{len(train_groups)} trips; "
            f"validation={len(val_idx)} rows/{len(val_groups)} trips"
        )

    return folds


def validate_required_columns(df):
    """Validate the candidate dataset before model training."""

    required = {
        TRIP_COL,
        OBS_COL,
        ARC_COL,
        TARGET_COL
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df[list(required)].isna().any().any():
        raise ValueError(
            "Required identifier or target columns contain missing values."
        )

    labels = set(df[TARGET_COL].unique())

    if not labels.issubset({0, 1}):
        raise ValueError(
            f"Target must contain only 0 and 1. Found: {labels}"
        )

    duplicates = df.duplicated(
        subset=[TRIP_COL, OBS_COL, ARC_COL]
    )

    if duplicates.any():
        raise ValueError(
            f"Found {duplicates.sum()} duplicate trip-observation-arc rows."
        )

    print("Dataset validation passed.")
