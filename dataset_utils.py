"""Shared helpers for the v2 dataset pipeline: paths, video ids, content crop, hashing and YOLO labels."""

import csv
import re
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT.parent / "data"
DATASETS_DIR = ROOT / "datasets"
WORK_DIR = DATASETS_DIR / "work"
VIDEOS_CSV = WORK_DIR / "videos.csv"

VIDEO_EXTS = {".mp4", ".mov"}

# Source folders inside RAW_DIR and the short prefix used in their video ids
SOURCES = {
    "dataset1": "d1",
    "tiktok croped": "tt",
    "archive": "ar",
    "archive test/raw_data/raw_data/data-crawl": "atr",
    "archive test/test/test": "att",
    "Not recognized": "nr",
}

VIDEO_FIELDS = [
    "video_id", "path", "source", "exercise", "w", "h", "fps", "frames", "dur",
    "crop", "orientation", "md5", "group", "use", "view", "split",
]

LEGACY_NAME = re.compile(r"^(\d+)_mp4-\d+_jpg\.rf\.")  # Roboflow frames of dataset1/<n>.mp4
FRAME_NAME = re.compile(r"^(.+)_f(\d+)$")  # frames written by extract_frames.py


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def natural_key(text: str) -> list:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", text)]


def split_source(rel: str) -> tuple[str, str] | None:
    """Split a path relative to RAW_DIR into (source prefix, path inside the source folder)."""
    for folder, prefix in sorted(SOURCES.items(), key=lambda kv: -len(kv[0])):
        if rel.startswith(folder + "/"):
            return prefix, rel[len(folder) + 1:]
    return None


def video_id(rel: str) -> str:
    """Stable id for a video path relative to RAW_DIR, e.g. 'atr_bench_press_14'."""
    prefix, inner = split_source(rel)
    parts = inner.split("/")
    stem = Path(parts[-1]).stem
    if len(parts) > 1 and not slug(stem).startswith(slug(parts[-2])):
        # 'squat/1.mp4' -> 'squat_n1', so it can't collide with 'squat/squat_1.mp4'
        stem = f"{parts[-2]} {'n' + stem if stem.isdigit() else stem}"
    return f"{prefix}_{slug(stem)}"


def frame_video_id(image_name: str) -> str | None:
    """Map a frame file name (legacy Roboflow or extract_frames.py) to its video id."""
    stem = Path(image_name).stem
    legacy = LEGACY_NAME.match(Path(image_name).name)
    if legacy:
        return f"d1_{legacy.group(1)}"
    frame = FRAME_NAME.match(stem)
    return frame.group(1) if frame else None


def orientation(w: int, h: int) -> str:
    ratio = w / h
    return "V" if ratio < 0.9 else ("S" if ratio <= 1.1 else "H")


BAR_THRESH = 25  # brightest pixel value still counted as a black bar


def content_box(col_bright: np.ndarray, row_bright: np.ndarray, min_share: float = 0.25,
                min_bar: float = 0.02) -> tuple[int, int, int, int]:
    """
    Find the picture inside black letterbox/pillarbox bars.

    col_bright / row_bright hold, per column / row, the share of sampled frames in which
    it has a pixel brighter than BAR_THRESH. Content must be bright in at least min_share
    of the frames, so a few transition or overlay frames don't hide the bars.
    A side is only cropped if its bar is at least min_bar of the frame size.
    Returns (x0, y0, x1, y1) with exclusive x1/y1.
    """
    w, h = len(col_bright), len(row_bright)
    cols = np.where(col_bright >= min_share)[0]
    rows = np.where(row_bright >= min_share)[0]
    if not len(cols) or not len(rows):
        return 0, 0, w, h
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0 = x0 if x0 >= min_bar * w else 0
    x1 = x1 if w - x1 >= min_bar * w else w
    y0 = y0 if y0 >= min_bar * h else 0
    y1 = y1 if h - y1 >= min_bar * h else h
    return x0, y0, x1, y1


def parse_crop(crop: str) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(v) for v in crop.split())
    return x0, y0, x1, y1


def dhash(gray: np.ndarray) -> int:
    """64-bit difference hash of a grayscale image."""
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def read_videos(path: Path = VIDEOS_CSV) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_videos(rows: list[dict], path: Path = VIDEOS_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: natural_key(r["video_id"]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=VIDEO_FIELDS)
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in VIDEO_FIELDS} for r in rows)


def read_yolo(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Read a YOLO label file as (cls, cx, cy, w, h) tuples with normalized coordinates."""
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5:
            boxes.append((int(parts[0]), *(float(v) for v in parts[1:])))
    return boxes


def write_yolo(path: Path, boxes: list[tuple[int, float, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in boxes))
