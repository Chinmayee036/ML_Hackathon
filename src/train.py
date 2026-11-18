import json
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import f1_score, log_loss
from .config import TRAIN_CSV, STUDENT_TEST_CSV, EXPERIMENTS, MODELS, TRIP_COL, TARGET_COL, N_SPLITS
from .data import load_candidate_data
from .models import build_models
from .validation import make_grouped_folds
from .metrics import top1_accuracy, candidate_miss_rate


def positive_probability(model, X):
    classes = list(model.classes_)
    positive_index = classes.index(1)
    return model.predict_proba(X)[:, positive_index]


def evaluate_model(name, estimator, train_df, feature_cols, folds, test_df=None):
    X = train_df[feature_cols]
    y = train_df[TARGET_COL].astype(int)
    groups = train_df[TRIP_COL]
    oof_prob = np.zeros(len(train_df), dtype=float)
    fold_f1 = []
    inference_seconds = []

    for train_idx, val_idx in folds:
        model = clone(estimator)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        start = time.perf_counter()
        probs = positive_probability(model, X.iloc[val_idx])
        inference_seconds.append(time.perf_counter() - start)
        oof_prob[val_idx] = probs
        preds = (probs >= 0.5).astype(int)
        fold_f1.append(f1_score(y.iloc[val_idx], preds, average="macro", zero_division=0))

    row = {
        "model": name,
        "cv_macro_f1": float(f1_score(y, oof_prob >= 0.5, average="macro", zero_division=0)),
        "mean_fold_macro_f1": float(np.mean(fold_f1)),
        "cv_log_loss": float(log_loss(y, np.clip(oof_prob, 1e-7, 1-1e-7), labels=[0, 1])),
        "cv_top1_accuracy": top1_accuracy(train_df, oof_prob),
        "candidate_miss_rate": candidate_miss_rate(train_df),
        "mean_fold_inference_seconds": float(np.mean(inference_seconds)),
    }

    if test_df is not None:
        X_test = test_df[feature_cols]
        y_test = test_df[TARGET_COL].astype(int)
        fitted = clone(estimator).fit(X, y)
        start = time.perf_counter()
        test_prob = positive_probability(fitted, X_test)
        elapsed = time.perf_counter() - start
        row.update({
            "student_test_macro_f1": float(f1_score(y_test, test_prob >= 0.5, average="macro", zero_division=0)),
            "student_test_log_loss": float(log_loss(y_test, np.clip(test_prob, 1e-7, 1-1e-7), labels=[0, 1])),
            "student_test_top1_accuracy": top1_accuracy(test_df, test_prob),
            "student_test_candidate_miss_rate": candidate_miss_rate(test_df),
            "student_test_inference_seconds": float(elapsed),
        })
    return row


def main():
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"{TRAIN_CSV} not found. Waiting for Person 1's aligned candidate CSV."
        )

    train_df, feature_cols = load_candidate_data(TRAIN_CSV)
    test_df = None
    if STUDENT_TEST_CSV.exists():
        test_df, test_features = load_candidate_data(STUDENT_TEST_CSV)
        if test_features != feature_cols:
            raise ValueError("Train/student-test feature columns do not match.")

    folds = make_grouped_folds(
        train_df[feature_cols], train_df[TARGET_COL],
        train_df[TRIP_COL], n_splits=N_SPLITS
    )

    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    results = []
    models = build_models()
    for name, estimator in models.items():
        print(f"\nEvaluating {name}...")
        result = evaluate_model(name, estimator, train_df, feature_cols, folds, test_df)
        results.append(result)
        print(result)

    results_df = pd.DataFrame(results).sort_values("cv_macro_f1", ascending=False)
    results_df.to_csv(EXPERIMENTS / "model_comparison.csv", index=False)

    best_name = results_df.iloc[0]["model"]
    best_model = clone(models[best_name]).fit(train_df[feature_cols], train_df[TARGET_COL].astype(int))
    joblib.dump(best_model, MODELS / "best_model.joblib")

    metadata = {
        "model_name": best_name,
        "feature_columns": feature_cols,
        "target_column": TARGET_COL,
        "group_column": TRIP_COL,
        "selection_metric": "grouped_cv_macro_f1",
        "cv_macro_f1": float(results_df.iloc[0]["cv_macro_f1"]),
    }
    (MODELS / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nSelected: {best_name}")
    print(f"Saved comparison to {EXPERIMENTS / 'model_comparison.csv'}")
    print(f"Saved model to {MODELS / 'best_model.joblib'}")


if __name__ == "__main__":
    main()
