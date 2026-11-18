"""Compare classifier-only and Viterbi results on student-test trips."""

import csv
from pathlib import Path

from task3.pipeline import LogisticScorer, ROOT, predict_trip


def main():
    scorer = LogisticScorer(ROOT / "round1/outputs/logistic_baseline.npz")
    split = ROOT / "Student folder/Student folder/student_test"
    results = []
    for trip in sorted(split.iterdir()):
        if not trip.is_dir():
            continue
        metrics = predict_trip(trip, scorer)["metrics"]
        result = {"trip_id": trip.name, **metrics}
        results.append(result)
        print(result, flush=True)
    with Path(__file__).with_name("student_test_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0])
        writer.writeheader()
        writer.writerows(results)
    n = sum(row["observations"] for row in results)
    moves = sum(row["transitions"] for row in results)
    for method in ("classifier", "decoded"):
        accuracy = sum(row[f"{method}_correct"] for row in results) / n
        continuity = sum(row[f"{method}_valid_transitions"] for row in results) / moves
        print(f"{method}: accuracy={accuracy:.4%} valid_transitions={continuity:.4%}")


if __name__ == "__main__":
    main()
