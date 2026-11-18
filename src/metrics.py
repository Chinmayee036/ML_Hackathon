import numpy as np
from sklearn.metrics import f1_score, log_loss, accuracy_score


def candidate_miss_rate(df, trip_col="trip_id", obs_col="obs_idx",
                        target_col="label"):
    """Fraction of GPS observations for which no candidate has a positive label."""
    per_obs = df.groupby([trip_col, obs_col])[target_col].max()
    return float((per_obs == 0).mean())


def top1_accuracy(df, probabilities, trip_col="trip_id",
                  obs_col="obs_idx", target_col="label"):
    """Choose the highest-scored candidate per observation and score its label."""
    temp = df[[trip_col, obs_col, target_col]].copy()
    temp["_score"] = np.asarray(probabilities)
    winners = temp.loc[temp.groupby([trip_col, obs_col])["_score"].idxmax()]
    return float(accuracy_score(winners[target_col], np.ones(len(winners), dtype=int)))


def classification_metrics(y_true, probabilities):
    probabilities = np.clip(np.asarray(probabilities), 1e-7, 1 - 1e-7)
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
    }
