
from .config import SAMPLE_CSV, TRIP_COL, OBS_COL, TARGET_COL
from .data import load_candidate_data


def main():
    df, features = load_candidate_data(SAMPLE_CSV)

    print("\nDATASET SHAPE:", df.shape)

    print("\nTRIPS:")
    print(df[TRIP_COL].nunique())

    print("\nOBSERVATIONS:")
    print(df[OBS_COL].nunique())

    print("\nLABEL DISTRIBUTION:")
    print(df[TARGET_COL].value_counts())

    print("\nCANDIDATES PER OBSERVATION:")
    print(df.groupby(OBS_COL).size().describe())

    print("\nPOSITIVE CANDIDATES PER OBSERVATION:")
    print(df.groupby(OBS_COL)[TARGET_COL].sum().value_counts())

    print("\nFEATURES:")
    print(features)

    print("\nDATASET VALIDATION PASSED!")


if __name__ == "__main__":
    main()
