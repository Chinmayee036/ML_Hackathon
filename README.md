# PathSnap – Person 1 (data, candidates, reference alignment)

## Run order
```
pip install -r requirements.txt
python scripts/01_build_candidates.py --root "<path>/Student folder"   # ~50 s -> outputs/candidates_*.csv.gz, trip_summary.csv
python scripts/02_eda.py            --root "<path>/Student folder"   # -> outputs/eda/*.csv, *.png
python scripts/03_folium_map.py     --root "<path>/Student folder" --partition train --trip 00000024
python scripts/04_radius_sweep.py   --root "<path>/Student folder"   # -> outputs/radius_sweep.csv
```

## Method (what to explain in viva)
1. **Projection**: per-trip equirectangular projection to metres (`pathsnap/geo.py`); degrees are never used as distance.
2. **Candidates**: each directed arc is densified (20 m samples) into a KD-tree; for each GPS fix take arcs within 50 m (grow radius if <3), keep the 10 closest by exact point-to-segment distance. Reverse-direction arcs are separate candidates.
3. **Reference alignment** (`pathsnap/align.py`): `.track` and `.route` rows do not correspond. We find a monotone map obs -> position on the ordered route minimising sum of (clipped) point-to-arc distance, with a small skip penalty (DP, O(N*M)). Label = aligned route arc id.
4. **Features** (`FEATURES` in `candidates.py`): distance, rank (ties broken by heading), distance minus min, projection fraction, arc length, heading cosine/angle (invalid if speed < 1 m/s), GPS speed, signed lateral offset, distance of neighbouring fixes (+-1, +-2) to the arc, distance to arc start/end, node degrees, #candidates, dt since previous fix.
5. **No label leakage**: reference arc is never injected into the candidate list; `build_trip(..., with_labels=False)` runs on unseen trips (use this in the API).

## Handoff table (one row = GPS fix x candidate arc)
`trip_id, obs_idx, t, lon, lat, x, y, arc_id` + `FEATURES` + `ref_arc` (aligned reference, -1 if unavailable) + `label` (1 if arc_id == ref_arc) (+ `align_ok` in the csv files).
For the API: `from pathsnap.candidates import build_trip; df, info, net, tr = build_trip(trip_dir, with_labels=False)`; `net` gives geometry/connectivity for Viterbi (`net.arcs`, `net.length`, `net.A/B`, `net.proj.inv`).

## Key results (see outputs/)
| | train (78 align_ok) | student_test (10) |
|---|---|---|
| candidate miss rate (R=50 m, K=10) | 1.04% | 0.21% |
| nearest-road top-1 accuracy | 79.98% | 81.82% |

## Data-quality findings
- Per-trip road network (different city/country per trip); 79/80 train networks have several connected components, no isolated nodes.
- ~97% of steps are 1 s; 60/80 train trips have some irregular gaps; max gap 79 s (trip 54).
- **Reference route unreliable for train trips 00000033 and 00000054** (GPS is on roads, but the route is 100+ m away through the middle of the trip; start/end match). Flagged `align_ok=False`; exclude from training and from reported metrics, and say so in the report.
- Reverse-direction arcs have identical geometry, so pure distance ties; the nearest-road baseline needs a heading tie-break (otherwise ~50%).
- Some reference routes have non-contiguous consecutive arcs (e.g. trip 00000002, 99.6%).

## Large data file (candidates_train)

`candidates_train.csv.gz` (128 MB) is over GitHub's 100 MB limit, so it is stored as split 7z parts in `outputs/candidates_train_parts/`.

To get the file: keep `candidates_train.7z.001` and `candidates_train.7z.002` in the same folder, then extract `.001` with 7-Zip. You get `candidates_train.csv.gz` (127,954,350 bytes, SHA256 `9b75a32e0289d722ac4ee0f923f27390e9c6ed681c365726088d0f93dd7c7631`).

Command line: `7z x outputs/candidates_train_parts/candidates_train.7z.001 -ooutputs`
