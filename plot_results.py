
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Project paths
ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = ROOT / "experiments" / "model_comparison.csv"
OUTPUT_DIR = ROOT / "experiments" / "graphs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_results():
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Results file not found: {RESULTS_FILE}\n"
            "Run python -m src.train first."
        )

    df = pd.read_csv(RESULTS_FILE)

    if "model" not in df.columns:
        raise ValueError("CSV must contain a 'model' column.")

    return df


def find_column(df, possible_names):
    """Find a metric column despite minor naming differences."""
    for name in possible_names:
        if name in df.columns:
            return name

    print(f"Warning: Could not find any of: {possible_names}")
    return None


def plot_comparison(df, cv_metric, test_metric,
                    title, ylabel, filename,
                    lower_better=False, log_scale=False):

    if cv_metric not in df.columns or test_metric not in df.columns:
        print(f"Skipping {title}: required columns not found.")
        return

    models = df["model"].astype(str)
    x = np.arange(len(models))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(
        x - width / 2,
        df[cv_metric],
        width,
        label="Grouped CV"
    )

    bars2 = ax.bar(
        x + width / 2,
        df[test_metric],
        width,
        label="Student Test"
    )

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")

    if log_scale:
        ax.set_yscale("log")

    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            value = bar.get_height()
            ax.annotate(
                f"{value:.3f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=8
            )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {filename}")


def plot_candidate_miss_rate(df):
    cv_col = find_column(df, ["candidate_miss_rate"])
    test_col = find_column(df, ["student_test_candidate_miss_rate"])

    if not cv_col or not test_col:
        return

    # These are dataset-level metrics, identical across models.
    train_rate = df[cv_col].iloc[0] * 100
    test_rate = df[test_col].iloc[0] * 100

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(
        ["Training Data", "Student Test"],
        [train_rate, test_rate]
    )

    ax.set_title("Candidate Miss Rate", fontsize=15, fontweight="bold")
    ax.set_ylabel("Observations with no positive candidate (%)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    for bar in bars:
        value = bar.get_height()
        ax.annotate(
            f"{value:.3f}%",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center"
        )

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "05_candidate_miss_rate.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    print("Saved: 05_candidate_miss_rate.png")


def main():
    df = load_results()

    print("\nMODEL RESULTS")
    print(df.to_string(index=False))

    # 1. Macro F1
    plot_comparison(
        df,
        "cv_macro_f1",
        "student_test_macro_f1",
        "Macro F1: Grouped CV vs Student Test",
        "Macro F1 (higher is better)",
        "01_macro_f1.png"
    )

    # 2. Top-1 Accuracy
    plot_comparison(
        df,
        "cv_top1_accuracy",
        "student_test_top1_accuracy",
        "Top-1 Accuracy: Grouped CV vs Student Test",
        "Accuracy (higher is better)",
        "02_top1_accuracy.png"
    )

    # 3. Log Loss
    plot_comparison(
        df,
        "cv_log_loss",
        "student_test_log_loss",
        "Log Loss: Grouped CV vs Student Test",
        "Log Loss (lower is better)",
        "03_log_loss.png"
    )

    # 4. Inference time
    plot_comparison(
        df,
        "mean_fold_inference_seconds",
        "student_test_inference_seconds",
        "Inference Time: Grouped CV vs Student Test",
        "Inference time in seconds (lower is better)",
        "04_inference_time.png",
        log_scale=True
    )

    # 5. Candidate miss rate
    plot_candidate_miss_rate(df)

    print("\nAll available graphs generated.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
