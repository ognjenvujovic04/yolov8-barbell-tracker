"""
Assign every video group in datasets/work/videos.csv to train/val/test (or test_hard).

The split is made per duplicate group, so copies of one video never end up in two splits,
and before any new labelling, so no teacher ever trains on val/test videos. Groups are
balanced per orientation, source and exercise. Existing assignments are frozen:
re-running only assigns groups that have no split yet.

Run from the project root in WSL (after build_index.py and the clip triage):
    python3 split_dataset.py
"""

import argparse
import random
from collections import Counter, defaultdict

from dataset_utils import VIDEOS_CSV, natural_key, read_videos, write_videos

SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
EXCLUDED_USES = {"skip"}


def group_stratum(rows: list[dict]) -> tuple[str, str, str]:
    """Orientation, source family and exercise/use of a group (square counts as vertical)."""
    primary = min(rows, key=lambda r: (r["use"] == "dup", natural_key(r["video_id"])))
    orient = "H" if primary["orientation"] == "H" else "V"
    family = "at" if primary["source"] in ("atr", "att") else primary["source"]
    return orient, family, primary["exercise"] or primary["use"]


def assign_splits(rows: list[dict], seed: int) -> int:
    """Fill the split column in place and return the number of newly assigned groups."""
    groups = defaultdict(list)
    for r in rows:
        if r["use"] not in EXCLUDED_USES and r["view"] != "front":
            groups[r["group"]].append(r)

    strata = defaultdict(list)
    counts = defaultdict(Counter)
    assigned = 0
    for group in sorted(groups, key=natural_key):
        members = groups[group]
        existing = {r["split"] for r in members if r["split"]}
        if len(existing) > 1:
            raise SystemExit(f"Group {group} is split across {sorted(existing)}; fix videos.csv by hand.")
        if any(r["use"] == "hard" for r in members):
            split = "test_hard"
        elif existing:
            split = existing.pop()
        else:
            strata[group_stratum(members)].append(group)
            continue
        for r in members:
            r["split"] = split
        counts[group_stratum(members)][split] += 1

    rng = random.Random(seed)
    for stratum in sorted(strata):
        todo = strata[stratum]
        rng.shuffle(todo)
        for group in todo:
            done = counts[stratum]
            total = sum(done[s] for s in SPLITS) + 1
            split = max(SPLITS, key=lambda s: (SPLITS[s] * total - done[s], SPLITS[s]))
            done[split] += 1
            for r in groups[group]:
                r["split"] = split
            assigned += 1
    return assigned


def print_summary(rows: list[dict]) -> None:
    table = defaultdict(lambda: [set(), 0, 0.0])
    for r in rows:
        if not r["split"]:
            continue
        key = ("H" if r["orientation"] == "H" else "V", r["split"], r["use"])
        table[key][0].add(r["group"])
        table[key][1] += 1
        table[key][2] += float(r["dur"]) / 60
    print(f"\n{'orient':6} {'split':9} {'use':7} {'groups':>6} {'videos':>6} {'minutes':>8}")
    for key in sorted(table):
        groups, videos, minutes = table[key]
        print(f"{key[0]:6} {key[1]:9} {key[2]:7} {len(groups):6d} {videos:6d} {minutes:8.1f}")
    unassigned = sum(1 for r in rows if not r["split"])
    print(f"\nWithout split (skip / front view): {unassigned} videos")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assign video groups in videos.csv to dataset splits.")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed for new groups (default: 0)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not VIDEOS_CSV.exists():
        raise SystemExit(f"{VIDEOS_CSV} not found. Run build_index.py first.")
    rows = read_videos()
    assigned = assign_splits(rows, args.seed)
    write_videos(rows)
    print(f"Assigned {assigned} new groups -> {VIDEOS_CSV}")
    print_summary(rows)


if __name__ == "__main__":
    main()
