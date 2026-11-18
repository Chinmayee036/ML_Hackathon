"""Local FastAPI entry point for the HMM and Viterbi route reconstruction."""

from functools import lru_cache
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from hmm_viterbi.demo import predict_demo
from hmm_viterbi.pipeline import LogisticScorer, ROOT, predict_trip

app = FastAPI(title="PathSnap HMM and Viterbi")
TRIPS = Path(os.environ.get("PATHSNAP_TRIPS", ROOT / "Student folder/Student folder/student_test"))
MODEL = Path(os.environ.get("PATHSNAP_MODEL", ROOT / "round1/outputs/logistic_baseline.npz"))
UI_ORIGIN = os.environ.get("PATHSNAP_UI_ORIGIN")
if UI_ORIGIN:
    app.add_middleware(CORSMiddleware, allow_origins=[UI_ORIGIN], allow_methods=["GET"])


@lru_cache(maxsize=1)
def scorer():
    return LogisticScorer(MODEL)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/trips")
def trips() -> dict[str, list[str]]:
    return {"trip_ids": sorted(
        path.name for path in TRIPS.glob("*")
        if path.is_dir() and (path / f"{path.name}.track").is_file()
    )}


@app.get("/trips/{trip_id}/prediction")
def predict(trip_id: str) -> dict:
    if trip_id == "demo":
        return predict_demo()
    if trip_id not in trips()["trip_ids"]:
        raise HTTPException(status_code=404, detail="Trip is unavailable")
    return predict_trip(TRIPS / trip_id, scorer())
