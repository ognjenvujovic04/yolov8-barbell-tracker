"""
Pre-label a frame batch with the teacher model for review in CVAT.

For datasets/work/batches/<batch>/ this writes two CVAT tasks' worth of input:

    upload_check/, cvat_<batch>_check.zip   frames to review carefully
    upload_quick/, cvat_<batch>_quick.zip   frames the teacher is confident about (skim)

Create a CVAT task from each upload_* folder, then "Upload annotations" with the
matching zip in the "Ultralytics YOLO Detection 1.0" format.

With --audit, the teacher instead checks already labelled frames of a v2 dataset and
packs the frames where it disagrees with the labels into datasets/work/batches/<name>/.
Its zip holds the current labels as 'barbell' and the teacher's extra boxes as 'check';
turn every 'check' box into 'barbell' or 'ignore', or delete it (build_dataset.py
refuses unresolved 'check' boxes). The CVAT project needs the labels barbell, ignore, check.

Run from the project root in WSL:
    python3 prelabel.py --batch b1_eval --weights runs/detect/teacher_t1/weights/best.pt
    python3 prelabel.py --audit datasets/v2/all.yaml --weights runs/detect/teacher_t3/weights/best.pt
"""

import argparse
import csv
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO

from dataset_utils import WORK_DIR, frame_video_id, read_yolo

BATCHES_DIR = WORK_DIR / "batches"
CVAT_LABELS = ["barbell", "ignore", "check"]
AUDIT_CONF = 0.5  # a teacher box this confident without a matching label is flagged
AUDIT_IOU = 0.5


def is_check(use: str, confs: list[float], check_conf: float) -> bool:
    """Frames where the teacher is likely wrong or the labelling rules need a decision."""
    if use == "neg":
        return bool(confs)  # a box on a negative video: missed barbell, background bar or false positive
    if not confs:
        return True  # a barbell video frame without a box
    return len(confs) >= 2 or min(confs) < check_conf


def write_cvat_zip(path: Path, labels: dict[str, list[str]]) -> None:
    """Annotations-only Ultralytics YOLO Detection zip; CVAT matches labels to task frames by name."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        names = "".join(f"  {i}: {name}\n" for i, name in enumerate(CVAT_LABELS))
        z.writestr("data.yaml", f"names:\n{names}path: .\ntrain: images/train\n")
        z.writestr("images/train/", "")  # the importer requires the folder, even when empty
        for stem, lines in sorted(labels.items()):
            z.writestr(f"labels/train/{stem}.txt", "".join(lines))


def prelabel(batch: str, weights: str, imgsz: int, conf: float, check_conf: float, device: str) -> None:
    batch_dir = BATCHES_DIR / batch
    with open(batch_dir / "frames.csv", newline="", encoding="utf-8") as f:
        frames = {r["stem"]: r for r in csv.DictReader(f)}
    model = YOLO(weights)
    parts = {"check": {}, "quick": {}}
    rows = []
    for result in model.predict(str(batch_dir / "images"), imgsz=imgsz, conf=conf, device=device,
                                stream=True, verbose=False):
        stem = Path(result.path).stem
        boxes = result.boxes
        confs = [float(c) for c in boxes.conf]
        lines = [f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for x, y, w, h in boxes.xywhn.tolist()]
        part = "check" if is_check(frames[stem]["use"], confs, check_conf) else "quick"
        parts[part][stem] = lines
        rows.append({"stem": stem, "use": frames[stem]["use"], "boxes": len(confs),
                     "min_conf": f"{min(confs):.3f}" if confs else "", "part": part})

    for part, labels in parts.items():
        upload = batch_dir / f"upload_{part}"
        if upload.exists():
            shutil.rmtree(upload)
        if not labels:
            continue
        upload.mkdir()
        for stem in labels:
            shutil.copy2(batch_dir / "images" / f"{stem}.jpg", upload / f"{stem}.jpg")
        write_cvat_zip(batch_dir / f"cvat_{batch}_{part}.zip", labels)

    with open(batch_dir / "prelabels.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stem", "use", "boxes", "min_conf", "part"])
        writer.writeheader()
        writer.writerows(rows)

    n_check, total = len(parts["check"]), len(rows)
    print(f"Batch '{batch}': {total} frames, {n_check} to check ({100 * n_check / max(total, 1):.0f}%), "
          f"{total - n_check} quick")
    for part in parts:
        if parts[part]:
            print(f"  CVAT task '{batch}_{part}': images {batch_dir / f'upload_{part}'}, "
                  f"annotations {batch_dir / f'cvat_{batch}_{part}.zip'}")


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between xyxy boxes a (n, 4) and b (m, 4)."""
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def xywhn_to_xyxy(boxes: list) -> np.ndarray:
    a = np.array([b[-4:] for b in boxes], dtype=float).reshape(-1, 4)
    return np.concatenate([a[:, :2] - a[:, 2:] / 2, a[:, :2] + a[:, 2:] / 2], axis=1)


def audit(data: Path, splits: list[str], name: str, weights: str, imgsz: int, conf: float, device: str) -> None:
    """Pack labelled frames where the teacher disagrees with the labels into a CVAT batch."""
    with open(data, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    batch_dir = BATCHES_DIR / name
    if batch_dir.exists():
        raise SystemExit(f"Batch '{name}' already exists: {batch_dir}")
    (batch_dir / "images").mkdir(parents=True)

    model = YOLO(weights)
    labels, rows = {}, []
    for split in splits:
        entries = cfg.get(split, [])
        for entry in entries if isinstance(entries, list) else [entries]:
            image_dir = data.parent / entry
            for result in model.predict(str(image_dir), imgsz=imgsz, conf=conf, device=device,
                                        stream=True, verbose=False):
                image = Path(result.path)
                gt = read_yolo(Path(str(image.parent).replace("images", "labels")) / f"{image.stem}.txt")
                pred = result.boxes.xywhn.tolist()
                confs = [float(c) for c in result.boxes.conf]
                iou = box_iou(xywhn_to_xyxy(pred), xywhn_to_xyxy(gt)) if pred and gt else np.zeros((len(pred), len(gt)))
                extra = [p for p, c, row in zip(pred, confs, iou) if c >= AUDIT_CONF and (row.max(initial=0) < AUDIT_IOU)]
                missing = int((iou.max(axis=0, initial=0) < AUDIT_IOU).sum()) if gt else 0
                if not extra and not missing:
                    continue
                shutil.copy2(image, batch_dir / "images" / image.name)
                labels[image.stem] = ([f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for _, cx, cy, w, h in gt]
                                      + [f"2 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for cx, cy, w, h in extra])
                rows.append({"stem": image.stem, "video_id": frame_video_id(image.name), "split": split,
                             "labels": len(gt), "extra_boxes": len(extra), "unmatched_labels": missing})

    with open(batch_dir / "frames.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stem", "video_id", "use", "split", "orientation"])
        writer.writeheader()
        writer.writerows({"stem": r["stem"], "video_id": r["video_id"], "use": "audit", "split": r["split"],
                          "orientation": ""} for r in rows)
    with open(batch_dir / "audit.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stem", "video_id", "split", "labels", "extra_boxes", "unmatched_labels"])
        writer.writeheader()
        writer.writerows(rows)
    if labels:
        write_cvat_zip(batch_dir / f"cvat_{name}.zip", labels)
    print(f"Audit '{name}': {len(rows)} frames flagged "
          f"({sum(r['extra_boxes'] > 0 for r in rows)} with extra teacher boxes, "
          f"{sum(r['unmatched_labels'] > 0 for r in rows)} with labels the teacher doesn't find)")
    print(f"  CVAT task '{name}': images {batch_dir / 'images'}, annotations {batch_dir / f'cvat_{name}.zip'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-label a frame batch with the teacher for CVAT review.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--batch", help="Batch name under datasets/work/batches/ to pre-label")
    mode.add_argument("--audit", type=Path, help="Dataset yaml whose labelled frames the teacher checks")
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        help="Splits to audit (default: train val)")
    parser.add_argument("--name", help="Audit batch name (default: audit_<teacher run>)")
    parser.add_argument("--weights", required=True, help="Teacher weights (.pt)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size (default: 640)")
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum box confidence (default: 0.25)")
    parser.add_argument("--check-conf", type=float, default=0.6,
                        help="Frames with a box below this confidence go to the check task (default: 0.6)")
    parser.add_argument("--device", default=None, help="Device: 0, cpu (default: GPU if available)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    if args.audit:
        name = args.name or f"audit_{Path(args.weights).parent.parent.name}"
        audit(args.audit, args.splits, name, args.weights, args.imgsz, args.conf, device)
    else:
        prelabel(args.batch, args.weights, args.imgsz, args.conf, args.check_conf, device)


if __name__ == "__main__":
    main()
