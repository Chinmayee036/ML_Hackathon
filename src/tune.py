import json
import joblib
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import TRAIN_CSV, EXPERIMENTS, MODELS, TRIP_COL, TARGET_COL, N_SPLITS, RANDOM_STATE
from .data import load_candidate_data
from .models import build_models
from .validation import make_grouped_folds
from .train import positive_probability


def make_estimator(model_name, trial):
    if model_name == "Logistic Regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=trial.suggest_float("C", 1e-3, 100, log=True),
                class_weight="balanced", max_iter=2500, random_state=RANDOM_STATE
            )
        )
    if model_name == "Gaussian Naive Bayes":
        return make_pipeline(
            StandardScaler(),
            GaussianNB(var_smoothing=trial.suggest_float("var_smoothing", 1e-12, 1e-7, log=True))
        )
    if model_name == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=trial.suggest_int("max_depth", 2, 30),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 30),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 15),
            class_weight="balanced", random_state=RANDOM_STATE
        )
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
            max_depth=trial.suggest_int("max_depth", 4, 30),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.7, 1.0]),
            class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE
        )
    return HistGradientBoostingClassifier(
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        max_iter=trial.suggest_int("max_iter", 75, 300),
        max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 7, 40),
        l2_regularization=trial.suggest_float("l2_regularization", 1e-5, 10, log=True),
        random_state=RANDOM_STATE
    )


def main():
    comparison_path = EXPERIMENTS / "model_comparison.csv"
    if not TRAIN_CSV.exists() or not comparison_path.exists():
        raise FileNotFoundError("Run Person 1's candidate handoff and `python -m src.train` first.")

    df, feature_cols = load_candidate_data(TRAIN_CSV)
    comparison = pd.read_csv(comparison_path)
    best_name = comparison.sort_values("cv_macro_f1", ascending=False).iloc[0]["model"]
    X, y, groups = df[feature_cols], df[TARGET_COL].astype(int), df[TRIP_COL]
    folds = make_grouped_folds(X, y, groups, N_SPLITS)

    def objective(trial):
        estimator = make_estimator(best_name, trial)
        scores = []
        for train_idx, val_idx in folds:
            model = clone(estimator)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            probs = positive_probability(model, X.iloc[val_idx])
            scores.append(f1_score(y.iloc[val_idx], probs >= 0.5,
                                   average="macro", zero_division=0))
        return sum(scores) / len(scores)

    # TPE is Optuna's Bayesian-style sequential sampler.
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=30)

    EXPERIMENTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    study.trials_dataframe().to_csv(EXPERIMENTS / "bayesian_trials.csv", index=False)
    best_info = {
        "model_name": best_name,
        "best_value_mean_grouped_cv_macro_f1": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
    }
    (EXPERIMENTS / "best_trial.json").write_text(json.dumps(best_info, indent=2))

    best_estimator = make_estimator(best_name, optuna.trial.FixedTrial(study.best_params))
    best_estimator.fit(X, y)
    joblib.dump(best_estimator, MODELS / "tuned_model.joblib")
    print(json.dumps(best_info, indent=2))
    print(f"Saved tuned model to {MODELS / 'tuned_model.joblib'}")


if __name__ == "__main__":
    main()
