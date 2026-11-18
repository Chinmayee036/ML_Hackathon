"""Align GPS observations to positions along the ordered reference route.

Why: one arc can hold many GPS points and short arcs can hold none, so row i of .track
does NOT correspond to row i of .route. We find a monotone (non-decreasing) assignment
obs i -> route position j(i) that minimises total point-to-arc distance, with a small
penalty for skipping route arcs. Dynamic programming, O(N*M).
"""
import numpy as np
from .geo import point_segment


def align_to_route(P, net, route, clip=60.0, skip_pen=1.0):
    """P: (N,2) projected GPS. Returns route position index per observation (N,)."""
    N, M = len(P), len(route)
    A, B = net.A[route], net.B[route]
    D = np.empty((N, M))
    for j in range(M):
        d, _, _ = point_segment(P, np.broadcast_to(A[j], P.shape), np.broadcast_to(B[j], P.shape))
        D[:, j] = np.minimum(d, clip)          # clip: robust to GPS outliers
    cost = D[0] + skip_pen * np.arange(M)      # may start slightly into the route
    back = np.zeros((N, M), dtype=np.int32)    # 0 = stay on arc, 1 = advance from j-1 (or skip chain)
    for i in range(1, N):
        stay = cost
        # best[j] = min(stay[j], best[j-1] + skip_pen)  (advance one or skip several arcs)
        best = stay.copy()
        came = np.zeros(M, dtype=np.int32)
        for j in range(1, M):
            if best[j - 1] + skip_pen < best[j]:
                best[j] = best[j - 1] + skip_pen
                came[j] = 1
        back[i] = came
        cost = best + D[i]
    cost = cost + skip_pen * (M - 1 - np.arange(M))
    j = int(np.argmin(cost))
    pos = np.empty(N, dtype=int)
    for i in range(N - 1, -1, -1):
        pos[i] = j
        # follow chain of advances backwards: came==1 means predecessor is j-1 (same step)
        while i > 0 and back[i][j] == 1:
            j -= 1
        # now j is the position at step i-1 (stay) -- loop continues
    return pos, D


def alignment_quality(D, pos):
    d = D[np.arange(len(pos)), pos]
    return {
        "align_med_m": float(np.median(d)),
        "align_p95_m": float(np.percentile(d, 95)),
        "align_frac_gt30m": float((d > 30).mean()),
        "route_arcs_with_no_obs": float(1 - len(np.unique(pos)) / D.shape[1]),
    }
