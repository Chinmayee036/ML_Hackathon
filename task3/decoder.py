"""Viterbi decoding for an ordered series of scored road candidates."""

from collections.abc import Callable, Mapping, Sequence
from math import inf, isfinite


def decode(
    emissions: Sequence[Mapping[int, float]],
    transition: Callable[[int, int, int], float],
) -> list[int | None]:
    """Return one arc per observation, using additive log scores.

    The transition callback receives the destination observation index and
    the previous/current arc IDs. Return -inf for an impossible transition.
    Empty observations and transitions with no reachable arc break the path.
    """
    path: list[int | None] = [None] * len(emissions)
    scores: dict[int, float] = {}
    history: list[tuple[int, dict[int, int | None]]] = []

    def finish_segment() -> None:
        if not scores:
            return
        arc = max(scores, key=scores.__getitem__)
        for index, parents in reversed(history):
            path[index] = arc
            arc = parents[arc]
        history.clear()

    for index, candidates in enumerate(emissions):
        valid = {arc: score for arc, score in candidates.items() if isfinite(score)}
        next_scores = {}
        parents = {}
        for current, emission in valid.items():
            best = -inf
            for previous, score in scores.items():
                total = score + transition(index, previous, current) + emission
                if total > best:
                    best = total
                    parents[current] = previous
            if best > -inf:
                next_scores[current] = best

        if not next_scores:
            finish_segment()
            next_scores = valid
            parents = dict.fromkeys(valid)

        scores = next_scores
        if scores:
            history.append((index, parents))

    finish_segment()
    return path
