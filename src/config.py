
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_PROCESSED = ROOT / "data" / "processed"
EXPERIMENTS = ROOT / "experiments"
MODELS = ROOT / "models"

TRAIN_CSV = DATA_PROCESSED / "candidates_train.csv"
STUDENT_TEST_CSV = DATA_PROCESSED / "candidates_student_test.csv"

SAMPLE_CSV = DATA_PROCESSED / "feature_table_sample_trip24.csv"

TRIP_COL = "trip_id"
OBS_COL = "obs_idx"
ARC_COL = "arc_id"
TARGET_COL = "label"

# Explicitly exclude IDs, reference answers and raw coordinates.
EXCLUDED_COLUMNS = {
    "trip_id",
    "obs_idx",
    "arc_id",
    "ref_arc",
    "label",
    "align_ok",
    "t",
    "lon",
    "lat",
    "x",
    "y",
}

N_SPLITS = 5
RANDOM_STATE = 42
