"""Predict a trip from GPS and road geometry, then score it server-side."""

from collections import defaultdict
from functools import lru_cache
from heapq import heappop, heappush
from math import ceil, inf
from pathlib import Path

import numpy as np

from round1.pathsnap.align import align_to_route
from round1.pathsnap.candidates import build_trip
from hmm_viterbi.decoder import decode


ROOT = Path(__file__).resolve().parents[1]


class LogisticScorer:
    def __init__(self, model_path):
        with np.load(model_path, allow_pickle=False) as model:
            self.features = model["features"].tolist()
            self.mean = model["mean"]
            self.scale = model["scale"]
            self.weights = model["weights"]

    def __call__(self, rows):
        values = (rows[self.features].to_numpy() - self.mean) / self.scale
        logits = values @ self.weights[1:] + self.weights[0]
        return -np.logaddexp(0, -logits)


def transitions(net, positions, projected, timestamps, beta):
    outgoing = defaultdict(list)
    for (start, end), length in zip(net.arcs.tolist(), net.length.tolist()):
        outgoing[start].append((end, length))
    starts, ends = net.arcs.T.tolist()
    lengths = net.length.tolist()
    displacement = np.linalg.norm(np.diff(projected, axis=0), axis=1)
    intervals = np.diff(timestamps)

    @lru_cache(maxsize=2048)
    def distances(start, limit):
        known = {start: 0.0}
        queue = [(0.0, start)]
        while queue:
            distance, node = heappop(queue)
            if distance != known[node]:
                continue
            for target, length in outgoing[node]:
                proposed = distance + length
                if proposed <= limit and proposed < known.get(target, inf):
                    known[target] = proposed
                    heappush(queue, (proposed, target))
        return known

    def transition(index, previous, current):
        elapsed = intervals[index - 1]
        if elapsed <= 0 or elapsed > 30:
            return -inf
        old = positions[index - 1][previous]
        new = positions[index][current]
        observed = displacement[index - 1]
        limit = 50 * ceil(max(80, 45 * elapsed + 30, 2 * observed) / 50)
        if previous == current and new >= old:
            travel = (new - old) * lengths[current]
        elif previous == current and (old - new) * lengths[current] <= 20:
            travel = 0.0
        else:
            bridge = distances(ends[previous], limit).get(starts[current], inf)
            travel = (1 - old) * lengths[previous] + bridge + new * lengths[current]
        if travel > limit:
            return -inf
        return -abs(travel - observed) / beta

    return transition


def predict_trip(directory, score, beta=20.0):
    directory = Path(directory)
    rows, _, net, raw = build_trip(directory, with_labels=False)
    gps = raw["track"]
    projected = net.proj.fwd(gps[:, 0], gps[:, 1])
    emissions = [{} for _ in gps]
    positions = [{} for _ in gps]
    for (index, arc, fraction), value in zip(
        rows[["obs_idx", "arc_id", "t_along"]].itertuples(index=False, name=None),
        score(rows),
    ):
        emissions[index][arc] = float(value)
        positions[index][arc] = fraction
    transition = transitions(net, positions, projected, gps[:, 2], beta)
    independent = [max(row, key=row.get) if row else None for row in emissions]
    decoded = decode(emissions, transition)

    def valid_moves(path):
        return int(sum(
            previous is not None and current is not None
            and transition(i, previous, current) > -inf
            for i, (previous, current) in enumerate(zip(path, path[1:]), 1)
        ))

    metrics = {
        "observations": len(gps),
        "transitions": len(gps) - 1,
        "classifier_valid_transitions": valid_moves(independent),
        "decoded_valid_transitions": valid_moves(decoded),
    }
    # Reference is read only after predictions are complete.
    reference_path = directory / f"{directory.name}.route"
    if reference_path.is_file():
        route = np.loadtxt(reference_path, dtype=int, ndmin=1)
        indices, _ = align_to_route(projected, net, route)
        reference = route[indices].tolist()
        for name, path in (("classifier", independent), ("decoded", decoded)):
            correct = sum(a == b for a, b in zip(reference, path))
            metrics[f"{name}_correct"] = correct
            metrics[f"{name}_top1"] = correct / len(gps)

    geometry = [
        {"arc_id": arc, "coordinates": raw["nodes"][net.arcs[arc]].tolist()}
        for i, arc in enumerate(decoded)
        if arc is not None and (i == 0 or decoded[i - 1] != arc)
    ]
    return {
        "trip_id": directory.name,
        "synthetic": False,
        "gps": gps[:, :2].tolist(),
        "classifier_arc_ids": independent,
        "decoded_arc_ids": decoded,
        "route": geometry,
        "metrics": metrics,
    }
