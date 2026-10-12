"""
Build the v2 YOLO datasets from reviewed CVAT exports.

Reads every Ultralytics YOLO Detection export in datasets/work/cvat/*.zip (later zips
override earlier ones for the same frame) and writes

    datasets/v2/{vertical,horizontal}/{train,val,test,test_hard}/{images,labels}
    datasets/v2/{vertical,horizontal,all}.yaml

Frames are routed by the split and orientation of their video in videos.csv. Legacy
1280x720 Roboflow frames are cropped to the video content (black bars removed).
Boxes labelled 'ignore' are painted grey and dropped, so the model is neither
rewarded nor punished for detecting them.

Run from the project root in WSL:
    python3 build_dataset.py
"""

import argparse
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from dataset_utils import (
    DATASETS_DIR, WORK_DIR, frame_video_id, natural_key, parse_crop, read_videos, write_yolo,
)

CVAT_DIR = WORK_DIR / "cvat"
BATCHES_DIR = WORK_DIR / "batches"
V2_DIR = DATASETS_DIR / "v2"
SPLITS = ["train", "val", "test", "test_hard"]
ORIENTATIONS = {"V": "vertical", "S": "vertical", "H": "horizontal"}
IGNORE_COLOR = (114, 114, 114)
MIN_BOX_PX = 2


def zip_names(z: zipfile.ZipFile) -> list[str]:
    names = yaml.safe_load(z.read("data.yaml"))["names"]
    if isinstance(names, dict):
        return [names[k] for k in sorted(names)]
    return list(names)


def load_exports() -> dict[str, dict]:
    """Collect labelled frames from all CVAT exports: stem -> {zip, image, labels, names}."""
    frames = {}
    zips = sorted(CVAT_DIR.glob("*.zip"), key=lambda p: (p.stat().st_mtime, p.name))
    for path in zips:
        with zipfile.ZipFile(path) as z:
            names = zip_names(z)
            images = {Path(n).stem: n for n in z.namelist() if n.startswith("images/") and not n.endswith("/")}
            for n in z.namelist():
                if n.startswith("labels/") and n.endswith(".txt"):
                    stem = Path(n).stem
                    frames[stem] = {
                        "zip": path, "image": images.get(stem), "labels": z.read(n).decode(), "names": names,
                    }
        print(f"Loaded {path.name}")
    return frames


def find_batch_image(stem: str) -> Path | None:
    """Annotation-only exports: the image lives in the batch folder it was extracted to."""
    for p in BATCHES_DIR.glob(f"*/images/{stem}.*"):
        return p
    return None


_open_zips: dict[Path, zipfile.ZipFile] = {}


def read_image(frame: dict, stem: str) -> np.ndarray:
    if frame["image"]:
        if frame["zip"] not in _open_zips:
            _open_zips[frame["zip"]] = zipfile.ZipFile(frame["zip"])
        data = np.frombuffer(_open_zips[frame["zip"]].read(frame["image"]), np.uint8)
    else:
        path = find_batch_image(stem)
        if path is None:
            raise FileNotFoundError(f"No image for frame {stem} (not in {frame['zip'].name} or batches/)")
        data = np.fromfile(path, np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def to_content(image: np.ndarray, video: dict) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop a full-size frame to the video content; frames that are already cropped pass through."""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = parse_crop(video["crop"])
    if (w, h) == (x1 - x0, y1 - y0):
        return image, (0, 0, w, h)
    if (w, h) == (int(video["w"]), int(video["h"])):
        return image[y0:y1, x0:x1], (x0, y0, x1, y1)
    raise ValueError(f"Frame size {w}x{h} matches neither video {video['video_id']} nor its crop")


def convert_labels(text: str, names: list[str], size: tuple[int, int], crop: tuple[int, int, int, int]):
    """Return (barbell boxes normalized to the crop, ignore boxes in crop pixels)."""
    w, h = size
    x0, y0, x1, y1 = crop
    cw, ch = x1 - x0, y1 - y0
    boxes, ignore = [], []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        name = names[int(parts[0])]
        cx, cy, bw, bh = (float(v) for v in parts[1:])
        bx0 = max((cx - bw / 2) * w - x0, 0)
        by0 = max((cy - bh / 2) * h - y0, 0)
        bx1 = min((cx + bw / 2) * w - x0, cw)
        by1 = min((cy + bh / 2) * h - y0, ch)
        if bx1 - bx0 < MIN_BOX_PX or by1 - by0 < MIN_BOX_PX:
            continue
        if name == "ignore":
            ignore.append((int(bx0), int(by0), int(round(bx1)), int(round(by1))))
        elif name == "barbell":
            boxes.append((0, (bx0 + bx1) / 2 / cw, (by0 + by1) / 2 / ch, (bx1 - bx0) / cw, (by1 - by0) / ch))
        elif name == "check":
            raise ValueError("Unresolved 'check' box from a label audit: make it barbell or ignore, or delete it")
        else:
            raise ValueError(f"Unknown label '{name}'")
    return boxes, ignore


def cap_frames(stems: list[str], limit: int) -> list[str]:
    """Keep at most `limit` frames, evenly spread over the video."""
    stems = sorted(stems, key=natural_key)
    if len(stems) <= limit:
        return stems
    return [stems[int(i)] for i in np.linspace(0, len(stems) - 1, limit)]


def write_yamls() -> list[Path]:
    def has_images(orient: str, split: str) -> bool:
        d = V2_DIR / orient / split / "images"
        return d.is_dir() and any(d.iterdir())

    written = []
    for name, orients in [("vertical", ["vertical"]), ("horizontal", ["horizontal"]),
                          ("all", ["vertical", "horizontal"])]:
        cfg = {"path": "."}
        for split in SPLITS:
            dirs = [f"{o}/{split}/images" for o in orients if has_images(o, split)]
            if dirs:
                cfg[split] = dirs[0] if len(dirs) == 1 else dirs
        if "train" not in cfg:
            continue
        cfg.update({"nc": 1, "names": ["barbell"]})
        out = V2_DIR / f"{name}.yaml"
        with open(out, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        written.append(out)
    return written


def build(max_train: int, max_eval: int) -> None:
    videos = {r["video_id"]: r for r in read_videos()}
    frames = load_exports()

    per_video = defaultdict(list)
    skipped = Counter()
    for stem in frames:
        vid = frame_video_id(stem)
        video = videos.get(vid)
        if video is None:
            skipped["unknown video"] += 1
        elif video["use"] in ("skip", "triage") or video["view"] == "front":
            skipped[f"video {video['use'] if video['view'] != 'front' else 'front view'}"] += 1
        elif not video["split"]:
            skipped["video without split"] += 1
        else:
            per_video[vid].append(stem)

    if V2_DIR.exists():
        shutil.rmtree(V2_DIR)

    stats = defaultdict(Counter)
    split_of_group = defaultdict(set)
    for vid in sorted(per_video, key=natural_key):
        video = videos[vid]
        split = video["split"]
        orient = ORIENTATIONS[video["orientation"]]
        limit = max_train if split == "train" else max_eval
        for stem in cap_frames(per_video[vid], limit):
            frame = frames[stem]
            image = read_image(frame, stem)
            h, w = image.shape[:2]
            image, crop = to_content(image, video)
            boxes, ignore = convert_labels(frame["labels"], frame["names"], (w, h), crop)
            for bx0, by0, bx1, by1 in ignore:
                cv2.rectangle(image, (bx0, by0), (bx1, by1), IGNORE_COLOR, thickness=-1)

            out = V2_DIR / orient / split
            (out / "images").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / "images" / f"{stem}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            write_yolo(out / "labels" / f"{stem}.txt", boxes)

            s = stats[(orient, split)]
            s["frames"] += 1
            s["boxes"] += len(boxes)
            s["empty"] += not boxes
            s["ignored"] += len(ignore)
            split_of_group[video["group"]].add(split)
        stats[(orient, split)]["videos"] += 1

    leaks = {g: s for g, s in split_of_group.items() if len(s) > 1}
    assert not leaks, f"Groups in more than one split: {leaks}"

    print(f"\n{'orientation':11} {'split':9} {'videos':>6} {'frames':>6} {'boxes':>6} {'empty':>6} {'ignored':>7}")
    for (orient, split) in sorted(stats):
        s = stats[(orient, split)]
        print(f"{orient:11} {split:9} {s['videos']:6d} {s['frames']:6d} {s['boxes']:6d} {s['empty']:6d} {s['ignored']:7d}")
    if skipped:
        print("\nSkipped frames: " + ", ".join(f"{k}: {v}" for k, v in sorted(skipped.items())))
    for path in write_yamls():
        print(f"Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build datasets/v2 from reviewed CVAT exports.")
    parser.add_argument("--max-train", type=int, default=60, help="Max frames per video in train (default: 60)")
    parser.add_argument("--max-eval", type=int, default=30, help="Max frames per video in val/test (default: 30)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build(args.max_train, args.max_eval)


if __name__ == "__main__":
    main()
