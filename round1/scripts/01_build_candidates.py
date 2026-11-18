"""Build candidate+feature tables for every trip in train and student_test.
Usage: python scripts/01_build_candidates.py --root "path/to/Student folder"
Outputs: outputs/candidates_{partition}.csv.gz, outputs/trip_summary.csv"""
import argparse, os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathsnap.io import list_trips, trip_dir, PARTITIONS
from pathsnap.candidates import build_trip

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--radius", type=float, default=50.0)
ap.add_argument("--max_cands", type=int, default=10)
ap.add_argument("--out", default="outputs")
ap.add_argument("--limit", type=int, help="Optional smoke-run trip limit per partition")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

infos = []
for part in PARTITIONS:
    frames = []
    for tid in list_trips(a.root, part)[:a.limit]:
        df, info, _, _ = build_trip(trip_dir(a.root, part, tid), a.radius, a.max_cands)
        info["partition"] = part
        df["align_ok"] = info["align_ok"]
        frames.append(df); infos.append(info)
    pd.concat(frames).to_csv(f"{a.out}/candidates_{part}.csv.gz", index=False)
s = pd.DataFrame(infos)
s.to_csv(f"{a.out}/trip_summary.csv", index=False)
for part in PARTITIONS:
    x = s[s.partition == part]
    w = x.n_obs
    print(f"{part}: trips={len(x)} obs={w.sum()} "
          f"miss_rate={(x.candidate_miss_rate*w).sum()/w.sum():.4f} "
          f"nearest_road_acc={(x.nearest_road_acc*w).sum()/w.sum():.4f}")
