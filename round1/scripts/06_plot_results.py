"""Render report figures from the completed PathSnap run. Requires Pillow."""

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
COLORS = ["#267784", "#3852AD"]
INK = "#192838"


def figure(title, subtitle):
    image = Image.new("RGB", (1100, 500), "white")
    draw = ImageDraw.Draw(image)
    label(draw, (52, 35), title, 35)
    label(draw, (52, 96), subtitle, 19)
    return image, draw


def label(draw, position, value, size=20):
    draw.text(position, value, fill=INK, font=ImageFont.load_default(size=size))


def plot_baseline():
    metrics = json.loads((OUTPUTS / "baseline_metrics.json").read_text())
    image, draw = figure(
        "Road candidate accuracy",
        f"{metrics['student_test_observations']:,} GPS observations across "
        f"{metrics['student_test_trips']} student-test trips",
    )
    for i, (name, key) in enumerate([
        ("Nearest road", "nearest_road_top1"),
        ("Logistic regression", "logistic_regression_top1"),
    ]):
        y = 176 + i * 108
        value = metrics[key]
        label(draw, (52, y + 10), name)
        draw.rectangle((275, y, 975, y + 50), fill="#EDF1F5")
        draw.rectangle((275, y, 275 + 700 * value, y + 50), fill=COLORS[i])
        label(draw, (990, y + 10), f"{value:.2%}")
    for tick in range(0, 101, 25):
        label(draw, (264 + 7 * tick, 365), f"{tick}%", 16)
    label(draw, (52, 438), "Top-1 accuracy against the aligned reference arcs, before Viterbi.", 18)
    image.save(OUTPUTS / "baseline_accuracy.png")


def plot_trip_sizes():
    with (OUTPUTS / "trip_summary.csv").open(newline="") as stream:
        trips = list(csv.DictReader(stream))
    bounds = [0, 1000, 2000, 4000, 8000, float("inf")]
    image, draw = figure("GPS observations per trip",
                         "Trip counts by trajectory length for the supplied data splits")
    groups = [
        [sum(row["partition"] == partition and lo <= int(row["n_obs"]) < hi
             for row in trips)
         for partition in ("train", "student_test")]
        for lo, hi in zip(bounds, bounds[1:])
    ]
    maximum = max(max(group) for group in groups)
    for i, (group, name) in enumerate(zip(
        groups, ["<1,000", "1,000-1,999", "2,000-3,999", "4,000-7,999", "8,000+"]
    )):
        x = 100 + i * 198
        for j, count in enumerate(group):
            left = x + 55 * j
            top = 370 - round(195 * count / maximum)
            if count:
                draw.rectangle((left, top, left + 39, 370), fill=COLORS[j])
            label(draw, (left + 8, top - 24), str(count), 16)
        label(draw, (x - 9, 388), name, 18)
    draw.rectangle((100, 448, 118, 466), fill=COLORS[0])
    label(draw, (128, 444), "Training", 18)
    draw.rectangle((260, 448, 278, 466), fill=COLORS[1])
    label(draw, (288, 444), "Student test", 18)
    image.save(OUTPUTS / "trip_sizes.png")


if __name__ == "__main__":
    plot_baseline()
    plot_trip_sizes()
