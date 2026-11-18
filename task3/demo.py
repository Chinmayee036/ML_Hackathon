"""Synthetic parallel-road example for testing the decoder and UI contract."""

import json
from math import inf

from task3.decoder import decode


def predict_demo() -> dict:
    geometry = {
        10: [[8.54, 49.35], [8.55, 49.35]],
        20: [[8.54, 49.3502], [8.55, 49.3502]],
    }
    gps = [[8.542, 49.35002], [8.545, 49.35018], [8.548, 49.35002]]
    emissions = [{10: -0.1, 20: -2.0}, {10: -1.2, 20: -0.1}, {10: -0.1, 20: -2.0}]

    def transition(index: int, previous: int, current: int) -> float:
        # The toy roads are disconnected within the observed window.
        return 0.0 if previous == current else -inf

    independent = [max(row, key=row.__getitem__) for row in emissions]
    decoded = decode(emissions, transition)
    route = []
    for index, arc in enumerate(decoded):
        if arc is not None and (index == 0 or decoded[index - 1] != arc):
            route.append({"arc_id": arc, "coordinates": geometry[arc]})
    return {
        "trip_id": "demo",
        "synthetic": True,
        "gps": gps,
        "classifier_arc_ids": independent,
        "decoded_arc_ids": decoded,
        "route": route,
        "metrics": {
            "classifier_switches": sum(a != b for a, b in zip(independent, independent[1:])),
            "decoded_switches": sum(a != b for a, b in zip(decoded, decoded[1:])),
        },
    }


if __name__ == "__main__":
    print(json.dumps(predict_demo(), indent=2))
