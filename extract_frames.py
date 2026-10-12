"""
Sample frames from source videos into a labelling batch.

Writes datasets/work/batches/<batch>/images/<video_id>_f<frame>.jpg (cropped to the video
content) and a frames.csv listing them. Videos are selected from datasets/work/videos.csv;
videos that are already labelled (in a CVAT export) or already in another batch are skipped.

Run from the project root in WSL, e.g.:
    python3 extract_frames.py --batch b1_eval --split val test
    python3 extract_frames.py --batch b1_h_train --split train --orientation H --limit 25
"""

import argparse
import csv
import random
import zipfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from dataset_utils import RAW_DIR, WORK_DIR, frame_video_id, natural_key, parse_crop, read_videos

BATCHES_DIR = WORK_DIR / "batches"
CVAT_DIR = WORK_DIR / "cvat"
EXTRACTABLE_USES = {"pos", "neg", "hard"}


def labelled_videos() -> set[str]:
    vids = set()
    for path in CVAT_DIR.glob("*.zip"):
        with zipfile.ZipFile(path) as z:
            vids.update(frame_video_id(Path(n).stem) for n in z.namelist() if n.startswith("labels/"))
    return vids


def batched_videos() -> set[str]:
    vids = set()
    for path in BATCHES_DIR.glob("*/frames.csv"):
        with open(path, newline="", encoding="utf-8") as f:
            vids.update(r["video_id"] for r in csv.DictReader(f) if r["use"] != "audit")
    return vids


def frame_indices(video: dict, fps: float, max_frames: int, neg_frames: int) -> list[int]:
    n = int(video["frames"])
    if n <= 0:
        return []
    if video["use"] == "neg":
        # A few frames from the middle 80%: negatives only need variety, not density
        return sorted({int(i) for i in np.linspace(0.1 * n, 0.9 * (n - 1), neg_frames)})
    count = max(1, min(max_frames, int(float(video["dur"]) * fps)))
    return sorted({int(i) for i in np.linspace(0, n - 1, count + 2)[1:-1]})


def extract_video(job: tuple[dict, list[int], Path]) -> list[dict]:
    video, indices, out_dir = job
    x0, y0, x1, y1 = parse_crop(video["crop"])
    cap = cv2.VideoCapture(str(RAW_DIR / video["path"]))
    rows = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        stem = f"{video['video_id']}_f{idx:05d}"
        cv2.imwrite(str(out_dir / f"{stem}.jpg"), frame[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append({
            "stem": stem, "video_id": video["video_id"], "use": video["use"],
            "split": video["split"], "orientation": video["orientation"],
        })
    cap.release()
    return rows


def select_videos(args: argparse.Namespace) -> list[dict]:
    rows = read_videos()
    done = labelled_videos() | batched_videos()
    selected = []
    for r in rows:
        if args.videos and r["video_id"] not in args.videos:
            continue
        if r["use"] not in args.use or r["view"] == "front" or not r["split"]:
            continue
        if args.split and r["split"] not in args.split:
            continue
        if args.orientation and r["orientation"] not in args.orientation:
            continue
        if r["video_id"] in done and not args.allow_repeat:
            continue
        selected.append(r)
    if args.limit and len(selected) > args.limit:
        selected = random.Random(args.seed).sample(selected, args.limit)
    return sorted(selected, key=lambda r: natural_key(r["video_id"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample video frames into a labelling batch.")
    parser.add_argument("--batch", required=True, help="Batch name (folder under datasets/work/batches/)")
    parser.add_argument("--split", nargs="+", choices=["train", "val", "test", "test_hard"], help="Only these splits")
    parser.add_argument("--orientation", nargs="+", choices=["V", "H", "S"], help="Only these orientations")
    parser.add_argument("--use", nargs="+", default=sorted(EXTRACTABLE_USES), choices=sorted(EXTRACTABLE_USES),
                        help="Only videos with these uses (default: pos neg hard)")
    parser.add_argument("--videos", nargs="+", help="Only these video ids")
    parser.add_argument("--limit", type=int, help="Random sample of at most this many videos")
    parser.add_argument("--seed", type=int, default=0, help="Seed for --limit (default: 0)")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second for pos/hard videos (default: 1)")
    parser.add_argument("--max-frames", type=int, default=30, help="Max frames per pos/hard video (default: 30)")
    parser.add_argument("--neg-frames", type=int, default=2, help="Frames per negative video (default: 2)")
    parser.add_argument("--allow-repeat", action="store_true", help="Include videos already labelled or batched")
    parser.add_argument("--workers", type=int, default=6, help="Parallel video readers (default: 6)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = BATCHES_DIR / args.batch / "images"
    if out_dir.exists():
        raise SystemExit(f"Batch '{args.batch}' already exists: {out_dir.parent}")

    videos = select_videos(args)
    if not videos:
        raise SystemExit("No videos match the selection.")
    out_dir.mkdir(parents=True)
    jobs = [(v, frame_indices(v, args.fps, args.max_frames, args.neg_frames), out_dir) for v in videos]

    rows = []
    with Pool(args.workers) as pool:
        for video_rows in pool.imap(extract_video, jobs):
            rows.extend(video_rows)
    with open(out_dir.parent / "frames.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stem", "video_id", "use", "split", "orientation"])
        writer.writeheader()
        writer.writerows(rows)

    uses = {u: sum(1 for r in rows if r["use"] == u) for u in sorted({r["use"] for r in rows})}
    print(f"Batch '{args.batch}': {len(videos)} videos, {len(rows)} frames {uses} -> {out_dir}")


if __name__ == "__main__":
    main()
