"""Candidate miss rate vs search radius / candidates kept (justifies radius=50 m, K=10).
Uses a fixed sample of align_ok train trips. Usage: python scripts/04_radius_sweep.py --root ..."""
import argparse, os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathsnap.io import trip_dir
from pathsnap.candidates import build_trip

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--n", type=int, default=12)
a = ap.parse_args()
s = pd.read_csv("outputs/trip_summary.csv"); s = s[(s.partition == "train") & s.align_ok]
ids = s.sample(a.n, random_state=0).trip_id.map(lambda x: f"{int(x):08d}").tolist()
rows = []
for R in (25, 50, 75, 100):
    for K in (5, 10, 15):
        miss = nn = n = cands = 0
        for tid in ids:
            _, info, _, _ = build_trip(trip_dir(a.root, "train", tid), R, K)
            miss += info["candidate_miss_rate"] * info["n_obs"]; nn += info["nearest_road_acc"] * info["n_obs"]
            cands += info["mean_cands"] * info["n_obs"]; n += info["n_obs"]
        rows.append(dict(radius_m=R, max_cands=K, miss_rate=miss / n, nearest_acc=nn / n, mean_cands=cands / n))
        print(rows[-1])
pd.DataFrame(rows).to_csv("outputs/radius_sweep.csv", index=False)
