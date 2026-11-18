"""Local metric projection + point-to-segment geometry (all distances in metres)."""
import numpy as np

R_EARTH = 6371008.8


class LocalProjection:
    """Equirectangular projection centred on the trip. Error is well under 0.1% over a
    city-sized extent, which is far below GPS noise. Avoids degree-based Euclidean distance."""

    def __init__(self, lon0, lat0):
        self.lon0, self.lat0 = lon0, lat0
        self.kx = np.deg2rad(1.0) * R_EARTH * np.cos(np.deg2rad(lat0))
        self.ky = np.deg2rad(1.0) * R_EARTH

    def fwd(self, lon, lat):
        return np.column_stack([(np.asarray(lon) - self.lon0) * self.kx,
                                (np.asarray(lat) - self.lat0) * self.ky])

    def inv(self, xy):
        xy = np.asarray(xy)
        return xy[:, 0] / self.kx + self.lon0, xy[:, 1] / self.ky + self.lat0


def point_segment(P, A, B):
    """Row-wise distance from points P to segments AB. Returns (dist, t in [0,1], foot)."""
    AB = B - A
    L2 = np.einsum("ij,ij->i", AB, AB)
    t = np.einsum("ij,ij->i", P - A, AB) / np.where(L2 > 0, L2, 1.0)
    t = np.clip(np.where(L2 > 0, t, 0.0), 0.0, 1.0)
    foot = A + AB * t[:, None]
    d = np.linalg.norm(P - foot, axis=1)
    return d, t, foot


def unit(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(n > 1e-9, v / np.where(n > 1e-9, n, 1), 0.0)
