"""Dataset exploration (report section 2.2). Writes CSV tables + PNG plots to outputs/eda/.
Usage: python scripts/02_eda.py --root "path/to/Student folder"
"""
import argparse, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathsnap.io import list_trips, trip_dir, load_trip, PARTITIONS
from pathsnap.geo import LocalProjection

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", default="outputs/eda")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

rows = []
for part in PARTITIONS:
    for tid in tqdm(list_trips(a.root, part), desc=part):
        d = trip_dir(a.root, part, tid)
        files = {e: os.path.join(d, f"{tid}.{e}") for e in ("track", "nodes", "arcs", "route")}
        t = load_trip(d)
        trk, nodes, arcs, route = t["track"], t["nodes"], t["arcs"], t["route"]
        lon, lat, ts = trk[:, 0], trk[:, 1], trk[:, 2]
        proj = LocalProjection(lon.mean(), lat.mean()); P = proj.fwd(lon, lat)
        dt = np.diff(ts)
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        n = len(nodes)
        g = coo_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(n, n))
        ncomp, lab = connected_components(g, directed=False)
        sizes = np.bincount(lab)
        used = np.zeros(n, bool); used[arcs[:, 0]] = True; used[arcs[:, 1]] = True
        route_ok = route is not None
        rows.append(dict(
            partition=part, trip_id=tid, n_files=sum(os.path.exists(f) for f in files.values()),
            size_kb=round(sum(os.path.getsize(f) for f in files.values() if os.path.exists(f)) / 1024, 1),
            n_obs=len(trk), n_nan=int(np.isnan(trk).sum()), duration_s=ts[-1] - ts[0],
            ts_monotonic=bool((dt > 0).all()), n_dt_not_1=int((dt != 1).sum()), frac_dt_1=float((dt == 1).mean()),
            max_dt_s=dt.max(), track_len_km=seg.sum() / 1000, max_step_m=seg.max(),
            lon_min=lon.min(), lon_max=lon.max(), lat_min=lat.min(), lat_max=lat.max(),
            lonlat_plausible=bool((np.abs(lon) <= 180).all() and (np.abs(lat) <= 90).all()),
            n_nodes=n, n_arcs=len(arcs), isolated_nodes=int((~used).sum()),
            n_components=ncomp, largest_comp_frac=sizes.max() / n,
            arc_ids_valid=bool(route_ok and route.min() >= 0 and route.max() < len(arcs)),
            route_len=len(route) if route_ok else np.nan,
            route_contiguous_frac=float((arcs[route[:-1], 1] == arcs[route[1:], 0]).mean()) if route_ok and len(route) > 1 else np.nan,
            obs_per_route_arc=len(trk) / len(route) if route_ok else np.nan))
df = pd.DataFrame(rows); df.to_csv(f"{a.out}/dataset_inventory.csv", index=False)

inv = df.groupby("partition").agg(trips=("trip_id", "count"), files=("n_files", "sum"), obs=("n_obs", "sum"),
      nan_values=("n_nan", "sum"), mean_obs=("n_obs", "mean"), median_duration_s=("duration_s", "median"),
      total_track_km=("track_len_km", "sum"), mean_nodes=("n_nodes", "mean"), mean_arcs=("n_arcs", "mean"),
      isolated_nodes=("isolated_nodes", "sum"), trips_multi_component=("n_components", lambda s: int((s > 1).sum())),
      trips_irregular_dt=("n_dt_not_1", lambda s: int((s > 0).sum()))).round(2)
inv.to_csv(f"{a.out}/partition_summary.csv"); print(inv.T)

# anomalies
an = df[(df.max_dt_s > 10) | (df.n_nan > 0) | (~df.ts_monotonic) | (df.route_contiguous_frac < 1) | (df.max_step_m > 200)]
an[["partition", "trip_id", "max_dt_s", "n_dt_not_1", "max_step_m", "route_contiguous_frac", "n_nan"]].to_csv(f"{a.out}/anomalous_trips.csv", index=False)
print("anomalous trips:", len(an))

# plots
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for part, c in zip(PARTITIONS, ("tab:blue", "tab:orange")):
    x = df[df.partition == part]
    ax[0, 0].hist(x.n_obs, bins=25, alpha=.7, color=c, label=part)
    ax[0, 1].hist(x.track_len_km, bins=25, alpha=.7, color=c, label=part)
    ax[0, 2].hist(x.n_arcs, bins=25, alpha=.7, color=c, label=part)
    ax[1, 0].scatter(x.n_obs, x.route_len, s=14, color=c, label=part)
    ax[1, 1].hist(x.frac_dt_1, bins=20, alpha=.7, color=c, label=part)
    ax[1, 2].scatter(x.n_nodes, x.n_components, s=14, color=c, label=part)
for k, (t_, xl) in enumerate([("GPS observations per trip", "obs"), ("Track length per trip", "km"), ("Road arcs per trip network", "arcs")]):
    ax[0, k].set_title(t_); ax[0, k].set_xlabel(xl); ax[0, k].legend()
ax[1, 0].set_title("Observations vs reference-route arcs"); ax[1, 0].set_xlabel("obs"); ax[1, 0].set_ylabel("route arcs")
ax[1, 1].set_title("Fraction of steps exactly 1 s"); ax[1, 1].set_xlabel("fraction")
ax[1, 2].set_title("Network size vs #components"); ax[1, 2].set_xlabel("nodes"); ax[1, 2].set_ylabel("components"); ax[1, 2].set_xscale("log")
plt.tight_layout(); plt.savefig(f"{a.out}/eda_overview.png", dpi=130); plt.close()

# trajectory shape plots for 6 trips (projected metres)
fig, ax = plt.subplots(2, 3, figsize=(15, 9))
for axx, (_, r) in zip(ax.ravel(), df[df.partition == "train"].sample(6, random_state=0).iterrows()):
    t = load_trip(trip_dir(a.root, "train", r.trip_id)); pr = LocalProjection(t["track"][:, 0].mean(), t["track"][:, 1].mean())
    N = pr.fwd(t["nodes"][:, 0], t["nodes"][:, 1]); P = pr.fwd(t["track"][:, 0], t["track"][:, 1])
    seg = np.stack([N[t["arcs"][:, 0]], N[t["arcs"][:, 1]]], 1)
    from matplotlib.collections import LineCollection
    axx.add_collection(LineCollection(seg, colors="lightgray", linewidths=.4))
    rt = t["route"]; axx.add_collection(LineCollection(seg[rt], colors="green", linewidths=1.5))
    axx.plot(P[:, 0], P[:, 1], "b.", ms=1.5); axx.set_xlim(P[:, 0].min() - 300, P[:, 0].max() + 300)
    axx.set_ylim(P[:, 1].min() - 300, P[:, 1].max() + 300); axx.set_aspect("equal"); axx.set_title(f"train {r.trip_id}  (grey=network, green=reference, blue=GPS)", fontsize=8)
plt.tight_layout(); plt.savefig(f"{a.out}/trajectories_train_sample.png", dpi=130); plt.close()

# candidate-set diagnostics from step 01 outputs (if present)
cp = "outputs/candidates_train.csv.gz"
if os.path.exists(cp):
    c = pd.read_csv(cp, usecols=["trip_id", "obs_idx", "dist_rank", "dist_m", "label", "align_ok"])
    c = c[c.align_ok]
    r = c[c.label == 1]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(c[c.dist_rank == 0].dist_m.clip(upper=60), bins=60); ax[0].set_title("Distance GPS -> nearest road (train, align_ok)"); ax[0].set_xlabel("m")
    ax[1].hist(r.dist_m.clip(upper=60), bins=60, color="green"); ax[1].set_title("Distance GPS -> reference arc (candidates that hit)"); ax[1].set_xlabel("m")
    plt.tight_layout(); plt.savefig(f"{a.out}/gps_noise_distances.png", dpi=130); plt.close()
print("EDA written to", a.out)
