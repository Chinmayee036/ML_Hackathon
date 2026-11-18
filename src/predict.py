import json
import joblib
import pandas as pd
from .config import MODELS, ID_COLUMNS


def score_candidates(candidates, model_path=None):
    """
    Score candidate rows for Person 3.

    Input must contain the same numeric feature columns used during training,
    plus trip_id, observation_index and arc_id metadata.
    """
    if model_path is None:
        model_path = MODELS / "tuned_model.joblib"
        if not model_path.exists():
            model_path = MODELS / "best_model.joblib"

    metadata_path = MODELS / "model_metadata.json"
    model = joblib.load(model_path)

    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        feature_cols = metadata["feature_columns"]
    else:
        raise FileNotFoundError(
            "model_metadata.json missing. Run training first or restore metadata."
        )

    missing = set(feature_cols) - set(candidates.columns)
    if missing:
        raise ValueError(f"Missing inference features: {sorted(missing)}")

    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(candidates[feature_cols])[:, positive_index]

    result = candidates.copy()
    result["candidate_score"] = probabilities
    return result
