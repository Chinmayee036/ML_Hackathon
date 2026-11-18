"""Candidate generation + feature engineering. Works with or without a reference route,
so the same function serves training, student-test scoring and unseen (API) trips."""
import numpy as np
import pandas as pd
from .io import load_trip
from .network import RoadNetwork
from .geo import point_segment, unit
from .align import align_to_route, alignment_quality

FEATURES = [
    "dist_m", "dist_rank", "dist_minus_min", "dist_ratio_min", "t_along", "arc_len",
    "heading_cos", "heading_absdeg", "gps_speed", "heading_valid", "lateral_signed",
    "d_prev", "d_next", "d_prev2", "d_next2", "nbr_mean_dist",
    "dist_to_end", "dist_to_start", "end_out_deg", "start_in_deg", "n_cands", "dt_prev",
]
ID_COLS = ["trip_id", "obs_idx", "t", "lon", "lat", "x", "y", "arc_id"]


def gps_motion(P, t):
    """Central-difference velocity (m/s) and unit heading per observation."""
    N = len(P)
    v = np.zeros((N, 2))
    if N > 1:
        lo = np.maximum(np.arange(N) - 1, 0)
        hi = np.minimum(np.arange(N) + 1, N - 1)
        dt = np.maximum(t[hi] - t[lo], 1e-6)
        v = (P[hi] - P[lo]) / dt[:, None]
    return v, np.linalg.norm(v, axis=1)


def build_trip(tdir, radius=50.0, max_cands=10, with_labels=True):
    """Return (candidate DataFrame, info dict). One row = (GPS observation, candidate arc)."""
    tr = load_trip(tdir, with_route=with_labels)
    net = RoadNetwork(tr["nodes"], tr["arcs"])
    trk = tr["track"]
    P = net.proj.fwd(trk[:, 0], trk[:, 1])
    t = trk[:, 2]
    N = len(P)
    vel, speed = gps_motion(P, t)
    ghead = unit(vel)
    dt_prev = np.r_[0.0, np.diff(t)]

    info = {"trip_id": tr["trip_id"], "n_obs": N}
    ref_arc = np.full(N, -1)
    if with_labels and tr["route"] is not None:
        route = tr["route"]
        pos, D = align_to_route(P, net, route)
        ref_arc = route[pos]
        info.update(alignment_quality(D, pos))
        info["route_len_arcs"] = len(route)
        info["align_ok"] = bool(info["align_med_m"] < 25 and info["align_frac_gt30m"] < 0.2)

    cand_ids = net.arcs_within(P, radius, min_arcs=3)
    rows_obs, rows_arc = [], []
    for i, ids in enumerate(cand_ids):
        d, _, _ = net.dist_to_arcs(np.broadcast_to(P[i], (len(ids), 2)), ids)
        keep = np.argsort(d, kind="stable")[:max_cands]
        # never drop the reference arc silently? NO: miss rate must reflect what the
        # real pipeline can do on unseen trips, so we do not inject the label.
        rows_obs.append(np.full(len(keep), i))
        rows_arc.append(ids[keep])
    oi = np.concatenate(rows_obs)
    ai = np.concatenate(rows_arc)
    Pc = P[oi]

    d, tt, foot = point_segment(Pc, net.A[ai], net.B[ai])
    df = pd.DataFrame({"obs_idx": oi, "arc_id": ai, "dist_m": d, "t_along": tt})
    g = df.groupby("obs_idx")["dist_m"]
    dmin = g.transform("min")
    df["dist_minus_min"] = df["dist_m"] - dmin
    df["dist_ratio_min"] = (df["dist_m"] + 1.0) / (dmin + 1.0)
    df["n_cands"] = g.transform("size")
    df["arc_len"] = net.length[ai]

    adir = net.dir[ai]
    gh = ghead[oi]
    cos = np.clip(np.einsum("ij,ij->i", gh, adir), -1, 1)
    valid = (speed[oi] > 1.0).astype(float)           # heading unreliable when ~stationary
    df["heading_valid"] = valid
    df["heading_cos"] = cos * valid
    df["heading_absdeg"] = np.where(valid > 0, np.degrees(np.arccos(cos)), 90.0)
    df["gps_speed"] = speed[oi]
    cross = adir[:, 0] * (Pc[:, 1] - net.A[ai][:, 1]) - adir[:, 1] * (Pc[:, 0] - net.A[ai][:, 0])
    df["lateral_signed"] = cross                      # which side of the arc the fix lies on

    # temporal-context cues: how far are the neighbouring fixes from this same arc?
    def nbr(shift):
        j = np.clip(oi + shift, 0, N - 1)
        dd, _, _ = point_segment(P[j], net.A[ai], net.B[ai])
        return dd
    df["d_prev"], df["d_next"] = nbr(-1), nbr(1)
    df["d_prev2"], df["d_next2"] = nbr(-2), nbr(2)
    df["nbr_mean_dist"] = (df["d_prev"] + df["d_next"] + df["d_prev2"] + df["d_next2"] + df["dist_m"]) / 5

    df["dist_to_start"] = np.linalg.norm(Pc - net.A[ai], axis=1)
    df["dist_to_end"] = np.linalg.norm(Pc - net.B[ai], axis=1)
    df["end_out_deg"] = net.out_deg[net.arcs[ai, 1]]
    df["start_in_deg"] = net.in_deg[net.arcs[ai, 0]]
    df["dt_prev"] = dt_prev[oi]

    # rank by distance; reverse-direction arcs have identical geometry (equal distance), so
    # break ties with heading agreement. Without this the "nearest road" baseline is capped ~50%.
    df["_d10"] = (df["dist_m"] * 10).round()
    order = df.sort_values(["obs_idx", "_d10", "heading_cos"], ascending=[True, True, False],
                           kind="stable")
    order["dist_rank"] = order.groupby("obs_idx").cumcount()
    df = order.drop(columns="_d10")

    df["trip_id"] = tr["trip_id"]
    df["t"] = t[oi]
    df["lon"], df["lat"] = trk[oi, 0], trk[oi, 1]
    df["x"], df["y"] = Pc[:, 0], Pc[:, 1]
    df["ref_arc"] = ref_arc[oi]
    df["label"] = (df["arc_id"].values == df["ref_arc"].values).astype(int) if with_labels else -1
    df = df[ID_COLS + FEATURES + ["ref_arc", "label"]]

    if with_labels and tr["route"] is not None:
        hit = df.groupby("obs_idx")["label"].max()
        info["candidate_miss_rate"] = float(1 - hit.mean())
        top1 = df[df.dist_rank == 0].set_index("obs_idx")["label"]
        info["nearest_road_acc"] = float(top1.mean())
    info["mean_cands"] = float(len(df) / N)
    return df, info, net, tr
