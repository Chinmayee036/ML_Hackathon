"""Write a mock API response JSON for one trip (for Person 4).

    python scripts/05_mock_response.py --root "C:/ML/Hackathon/Student folder"
    python scripts/05_mock_response.py --root "..." --partition student_test --trip <id>
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathsnap.io import trip_dir  # noqa: E402
from pathsnap.export import build_response, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="path to 'Student folder'")
    ap.add_argument("--partition", default="train")
    ap.add_argument("--trip", default="00000024")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = a.out or os.path.join("outputs", f"mock_response_{a.trip}.json")
    resp = build_response(trip_dir(a.root, a.partition, a.trip))
    save_json(resp, out)
    print(f"wrote {out}: {resp['n_obs']} fixes, {len(resp['matched_path'])} arcs in path, "
          f"{resp['n_unmatched']} unmatched")


if __name__ == "__main__":
    main()
