"""Loading raw PathSnap trips. Row index of .nodes = node id, row index of .arcs = arc id."""
import os
import numpy as np

PARTITIONS = ("train", "student_test")


def trip_dir(root, partition, trip_id):
    return os.path.join(root, partition, trip_id)


def list_trips(root, partition):
    d = os.path.join(root, partition)
    return sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)))


def _load(path, dtype=float, ncols=None):
    # files are whitespace separated, some track lines end with \r
    a = np.loadtxt(path, dtype=dtype, ndmin=2)
    if ncols is not None and a.shape[1] != ncols:
        raise ValueError(f"{path}: expected {ncols} columns, got {a.shape[1]}")
    return a


def load_trip(tdir, with_route=True):
    """Return dict with track(lon,lat,t), nodes(lon,lat), arcs(from,to), route(arc ids or None)."""
    tid = os.path.basename(os.path.normpath(tdir))
    p = os.path.join(tdir, tid)
    out = {
        "trip_id": tid,
        "track": _load(p + ".track", float, 3),
        "nodes": _load(p + ".nodes", float, 2),
        "arcs": _load(p + ".arcs", int, 2),
        "route": None,
    }
    rp = p + ".route"
    if with_route and os.path.exists(rp):
        out["route"] = _load(rp, int, 1)[:, 0]
    return out
