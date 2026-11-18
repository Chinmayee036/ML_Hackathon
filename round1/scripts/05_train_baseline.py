"""Fit a NumPy logistic baseline and evaluate it on separate student-test trips."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.candidates import FEATURES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("round1/outputs"))
    args = parser.parse_args()
    train, test = [
        pd.read_csv(args.data / f"candidates_{split}.csv.gz", dtype={"trip_id": str})
        for split in ("train", "student_test")
    ]
    assert set(train.trip_id).isdisjoint(test.trip_id)
    x = train[FEATURES].to_numpy(dtype=float)
    target = train.label.to_numpy()
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1
    x = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    weights = np.zeros(x.shape[1])
    for _ in range(150):
        probability = 1 / (1 + np.exp(-np.clip(x @ weights, -50, 50)))
        gradient = x.T @ (probability - target) / len(x)
        gradient[1:] += 0.001 * weights[1:]
        weights -= 0.3 * gradient

    keys = ["trip_id", "obs_idx"]
    scores = ((test[FEATURES].to_numpy() - mean) / scale) @ weights[1:] + weights[0]
    scored = test.assign(score=scores)
    best = scored.groupby(keys).score.idxmax()
    positive = scored.groupby(keys).label.max()
    nearest = scored.loc[scored.dist_rank == 0, "label"]
    metrics = {
        "train_trips": int(train.trip_id.nunique()),
        "student_test_trips": int(test.trip_id.nunique()),
        "training_candidate_rows": len(train),
        "student_test_observations": len(positive),
        "candidate_miss_rate": 1 - float(positive.mean()),
        "nearest_road_top1": float(nearest.mean()),
        "logistic_regression_top1": float(scored.loc[best, "label"].mean()),
    }
    np.savez(args.data / "logistic_baseline.npz", features=FEATURES,
             mean=mean, scale=scale, weights=weights)
    report = json.dumps(metrics, indent=2)
    (args.data / "baseline_metrics.json").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
