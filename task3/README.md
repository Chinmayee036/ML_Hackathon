# PathSnap Task 3

Run these commands from the workspace root using a Python environment with
NumPy, pandas, FastAPI, and Uvicorn.

```powershell
python -m pip install -r task3/requirements.txt
python -m task3.evaluate
python -m uvicorn task3.api:app --host 0.0.0.0 --port 8000
```

The evaluation writes per-trip metrics to `task3/student_test_results.csv`.
On 19,160 GPS observations the logistic classifier matched 84.16% of the
aligned reference arcs, and Viterbi matched 90.91%. The share of consecutive
pairs accepted by the transition rule went from 93.62% to 99.95%.

## Scoring and decoding

The decoder uses the log of the saved logistic model's positive-class probability
as the emission score. It compares the straight-line GPS displacement with the
directed network distance between projected candidate points. A bounded shortest
path finds connections between arcs, and the score penalizes displacement
mismatch. The prototype uses a 20 m penalty scale, permits up to 20 m of
backward projection jitter on the same arc, and starts a new segment after a
gap longer than 30 seconds or if no transition remains possible. Reference
routes are read after prediction to compute accuracy.

## Laptop 2 handoff

Open `http://<laptop-1-ip>:8000/docs` from Laptop 2. If the browser UI calls the
API directly, set `PATHSNAP_UI_ORIGIN` on Laptop 1 to the UI's full origin
including its port before starting Uvicorn.

- `GET /health` checks connectivity.
- `GET /trips` returns available trip IDs from the configured directory.
- `GET /trips/{trip_id}/prediction` returns GPS observations, both predicted
  arc sequences, geometry for the decoded arcs in longitude-latitude order, and
  aggregate metrics. Draw each geometry entry as its own segment.

The default trip directory contains the ten student-test trips. Set
`PATHSNAP_TRIPS` to the organiser-provided directory at final evaluation.
Set `PATHSNAP_MODEL` to relocate the baseline artifact. The current scorer
loads the NumPy logistic model. Integrating Person 2's selected estimator
requires its model artifact, feature order, and matching runtime dependencies.
The code and experiments in this starter were AI-assisted.
