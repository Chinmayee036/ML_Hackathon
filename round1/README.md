# PathSnap first evaluation

The preprocessing code in this folder comes from Person 1's GitHub commit
`cd0c345`. This local copy uses a NumPy grid index when SciPy is unavailable.
The grid applies an exact distance filter to the sampled road geometry.

## What to show

- Open the generated `trip_summary.csv` and candidate tables. Each row in a
  candidate table describes one GPS observation and one possible road arc.
  Show `dist_m`, `heading_cos`, and `label`. The label comes from a monotone
  alignment to the ordered reference route.
- Explain why logistic regression was used for the first run. It gives an
  interpretable score from standardised distance, heading, and context
  features, and provides a baseline for later model comparison.
- Open `baseline_metrics.json`. Top-1 measures whether the highest-scoring
  candidate matches the aligned reference arc. Observations whose reference
  arc is missing from the candidate set count as misses.

## Reproduce

From the workspace root, use a Python environment with NumPy and pandas.

```powershell
python round1/scripts/01_build_candidates.py --root 'Student folder/Student folder' --out round1/outputs
python round1/scripts/05_train_baseline.py --data round1/outputs
```

The preliminary run used `--limit 3 --out round1/smoke` and fitted on three
training trips. Evaluation used three different student-test trips.
The training script saves the scaler and model weights in
`logistic_baseline.npz` and metrics in `baseline_metrics.json`.
The local dependency fallback and baseline training script were AI-assisted.
