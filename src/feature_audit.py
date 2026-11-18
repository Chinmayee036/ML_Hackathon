
from src.data import load_candidate_data
from src.config import SAMPLE_CSV, TRIP_COL, OBS_COL, TARGET_COL

def main():
    df, features = load_candidate_data(SAMPLE_CSV)

    print("\n========== FEATURE AUDIT ==========")

    print("\nFeature count:", len(features))

    print("\nClass distribution:")
    print(df[TARGET_COL].value_counts(normalize=True))

    print("\nFeature statistics:")
    print(df[features].describe().T)

    print("\nHighly correlated features:")
    corr = df[features].corr().abs()

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            if corr.iloc[i, j] > 0.95:
                print(
                    features[i],
                    "<->",
                    features[j],
                    round(corr.iloc[i, j], 3)
                )

    print("\nPositive labels per observation:")
    print(
        df.groupby([TRIP_COL, OBS_COL])[TARGET_COL]
        .sum().value_counts()
    )

    print("\n========== AUDIT COMPLETE ==========")


if __name__ == "__main__":
    main()
