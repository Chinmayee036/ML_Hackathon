"""Export helpers: arc -> lat/lon geometry and a mock API response.

The mock uses the nearest-road baseline as a placeholder prediction, so Person 4 can build
and test the API / map without waiting for the real model. Swap `predict_arcs` for the
real model's output when it exists; everything else stays the same.
"""
import json
import os

import numpy as np

from .candidates import build_trip
from .geo import point_segment


def arc_to_lonlat(net, arc_id):
    """Arc geometry as [[lon, lat], [lon, lat]] (start node, end node)."""
    xy = np.vstack([net.A[arc_id], net.B[arc_id]])
    lon, lat = net.proj.inv(xy)
    return [[round(float(lo), 6), round(float(la), 6)] for lo, la in zip(lon, lat)]


def snap_to_arc(net, xy, arc_id):
    """Project a GPS point (x, y in metres) onto an arc.
    Returns (snapped_lon, snapped_lat, distance_m)."""
    P = np.asarray(xy, float)[None, :]
    d, _, foot = point_segment(P, net.A[[arc_id]], net.B[[arc_id]])
    lon, lat = net.proj.inv(foot)
    return float(lon[0]), float(lat[0]), float(d[0])


def predict_arcs(df, n_obs):
    """Placeholder prediction: the rank-0 (nearest, heading tie-break) candidate per fix.
    Returns a DataFrame indexed 0..n_obs-1; fixes with no candidate have NaN arc_id."""
    top = df[df["dist_rank"] == 0].drop_duplicates("obs_idx").set_index("obs_idx")
    return top.reindex(range(n_obs))


def build_response(tdir):
    """Run the pipeline on one trip folder (no labels needed) and return a JSON-ready dict."""
    df, info, net, tr = build_trip(tdir, with_labels=False)
    n = info["n_obs"]
    top = predict_arcs(df, n)

    points, path = [], []
    for i in range(n):
        row = top.iloc[i]
        if np.isnan(row["arc_id"]):
            points.append({"obs_idx": i, "arc_id": None, "snapped": None})
            continue
        arc = int(row["arc_id"])
        slon, slat, d = snap_to_arc(net, (row["x"], row["y"]), arc)
        points.append({
            "obs_idx": i,
            "t": float(row["t"]),
            "gps": [round(float(row["lon"]), 6), round(float(row["lat"]), 6)],
            "arc_id": arc,
            "snapped": [round(slon, 6), round(slat, 6)],
            "dist_m": round(d, 2),
        })
        if not path or path[-1] != arc:      # collapse consecutive repeats
            path.append(arc)

    return {
        "trip_id": info["trip_id"],
        "method": "nearest_road_baseline (MOCK - replace with model output)",
        "n_obs": n,
        "n_unmatched": int(sum(p["arc_id"] is None for p in points)),
        "points": points,
        "matched_path": [{"arc_id": a, "geometry": arc_to_lonlat(net, a)} for a in path],
    }


def save_json(resp, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(resp, f)
    return out_path
