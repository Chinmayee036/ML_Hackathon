"""Road network wrapper: projected geometry, spatial index, degree info."""
import numpy as np
from scipy.spatial import cKDTree
from .geo import LocalProjection, point_segment


class RoadNetwork:
    def __init__(self, nodes_lonlat, arcs, proj=None, step=20.0):
        self.arcs = arcs
        if proj is None:
            proj = LocalProjection(nodes_lonlat[:, 0].mean(), nodes_lonlat[:, 1].mean())
        self.proj = proj
        self.nodes_xy = proj.fwd(nodes_lonlat[:, 0], nodes_lonlat[:, 1])
        self.A = self.nodes_xy[arcs[:, 0]]
        self.B = self.nodes_xy[arcs[:, 1]]
        d = self.B - self.A
        self.length = np.linalg.norm(d, axis=1)
        self.dir = d / np.where(self.length[:, None] > 0, self.length[:, None], 1.0)
        n = len(nodes_lonlat)
        self.out_deg = np.bincount(arcs[:, 0], minlength=n)
        self.in_deg = np.bincount(arcs[:, 1], minlength=n)
        self._build_index(step)

    def _build_index(self, step):
        # densify every arc into sample points <= `step` metres apart; KD-tree over samples.
        # Any arc within R of a point has a sample within R + step/2 of it.
        self.step = step
        k = np.maximum(np.ceil(self.length / step).astype(int), 1) + 1
        arc_of = np.repeat(np.arange(len(k)), k)
        offs = np.cumsum(k) - k
        j = np.arange(len(arc_of)) - np.repeat(offs, k)
        t = j / np.repeat(k - 1, k)
        pts = self.A[arc_of] + (self.B[arc_of] - self.A[arc_of]) * t[:, None]
        self.sample_arc = arc_of
        self.tree = cKDTree(pts)

    def arcs_within(self, xy, radius, min_arcs=3, max_radius=400.0):
        """For each point -> unique arc ids within radius (radius grows until min_arcs found)."""
        out = []
        for p in xy:
            r = radius
            while True:
                idx = self.tree.query_ball_point(p, r + self.step / 2)
                ids = np.unique(self.sample_arc[idx]) if len(idx) else np.empty(0, int)
                if len(ids) >= min_arcs or r >= max_radius:
                    break
                r *= 2
            out.append(ids)
        return out

    def dist_to_arcs(self, P, arc_ids):
        d, t, foot = point_segment(P, self.A[arc_ids], self.B[arc_ids])
        return d, t, foot
