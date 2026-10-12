"""
Pre-label a frame batch with the teacher model for review in CVAT.

For datasets/work/batches/<batch>/ this writes two CVAT tasks' worth of input:

    upload_check/, cvat_<batch>_check.zip   frames to review carefully
    upload_quick/, cvat_<batch>_quick.zip   frames the teacher is confident about (skim)

Create a CVAT task from each upload_* folder, then "Upload annotations" with the
matching zip in the "Ultralytics YOLO Detection 1.0" format.

Run from the project root in WSL:
    python3 prelabel.py --batch b1_eval --weights runs/detect/teacher_t1/weights/best.pt
"""

import argparse
import csv
import shutil
import zipfile
from pathlib import Path

import torch
from ultralytics import YOLO

from dataset_utils import WORK_DIR

BATCHES_DIR = WORK_DIR / "batches"
CVAT_LABELS = ["barbell", "ignore"]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-label a frame batch with the teacher for CVAT review.")
    parser.add_argument("--batch", required=True, help="Batch name under datasets/work/batches/")
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
    prelabel(args.batch, args.weights, args.imgsz, args.conf, args.check_conf, device)


if __name__ == "__main__":
    main()
