"""
Evaluate a barbell detector on one split of a v2 dataset.

Reports Precision, Recall, mAP@0.5, mAP@0.5:0.95 and the confidence threshold with the
best F1 score (use it as the app's detection threshold). Validation runs at --imgsz with
rectangular batches, so a 9:16 frame is evaluated at the same scale as the 384x224 export.

Run from the project root in WSL:
    python test.py --model runs/detect/student_vertical/weights/best.pt --data datasets/v2/vertical.yaml
    python test.py --model best.pt --data datasets/v2/vertical.yaml --split test_hard --imgsz 320
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "best.pt"
DEFAULT_DATA = ROOT / "datasets" / "v2" / "vertical.yaml"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_eval_yaml(data: Path, split: str) -> tuple[Path, list[Path]]:
    """Write <data>.eval.yaml whose 'test' entry points at the requested split."""
    with open(data, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if split not in cfg:
        raise SystemExit(f"Split '{split}' not in {data} (available: {[k for k in cfg if k not in ('path', 'nc', 'names')]})")

    root = data.parent.resolve()
    entries = cfg[split] if isinstance(cfg[split], list) else [cfg[split]]
    image_dirs = [root / e for e in entries]
    config = {"path": str(root), "train": cfg["train"], "val": cfg.get("val", cfg["train"]),
              "test": cfg[split], "nc": cfg["nc"], "names": cfg["names"]}

    out = data.with_name(f"{data.stem}.eval.yaml")
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return out, image_dirs


def count_images(image_dirs: list[Path]) -> int:
    return sum(1 for d in image_dirs for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def best_f1_threshold(box) -> tuple[float, float]:
    """Confidence with the highest F1 (averaged over classes) and that F1."""
    f1 = np.asarray(box.f1_curve).mean(axis=0)
    i = int(f1.argmax())
    return float(box.px[i]), float(f1[i])


def print_metrics(metrics, split: str, model_path: Path, imgsz: int, n_images: int) -> None:
    box = metrics.box
    conf, f1 = best_f1_threshold(box)
    print("\n" + "=" * 60)
    print(f"Barbell detector — {split} set evaluation")
    print("=" * 60)
    print(f"Model:      {model_path}")
    print(f"Image size: {imgsz} (long side, rectangular batches)")
    print(f"Images:     {n_images}")
    print("-" * 60)
    print(f"Precision (P):     {box.mp:.4f}")
    print(f"Recall (R):        {box.mr:.4f}")
    print(f"mAP@0.5:           {box.map50:.4f}")
    print(f"mAP@0.5:0.95:      {box.map:.4f}")
    print(f"Best F1:           {f1:.4f} at confidence {conf:.3f}  <- app threshold")
    print("=" * 60)
    if metrics.speed:
        print(
            f"Inference: {metrics.speed.get('inference', 0):.1f} ms/img  "
            f"(preprocess {metrics.speed.get('preprocess', 0):.1f}, "
            f"postprocess {metrics.speed.get('postprocess', 0):.1f})"
        )
    print(f"Results saved to: {metrics.save_dir}")


def evaluate(model_path: Path, data: Path, split: str, imgsz: int, device: str | None) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if model_path.suffix != ".pt":
        raise SystemExit("Metrics need a .pt model: Ultralytics can't validate rectangular TFLite exports.")

    data_yaml, image_dirs = build_eval_yaml(data, split)
    n_images = count_images(image_dirs)
    if n_images == 0:
        raise RuntimeError(f"No images found in {image_dirs}")

    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {model_path}")
    print(f"Device: {device}")
    print(f"{split} images: {n_images}")
    print(f"Data config: {data_yaml}")

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=imgsz,
        device=device,
        verbose=True,
        plots=True,
        save_json=True,
    )

    print_metrics(metrics, split, model_path, imgsz, n_images)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a barbell YOLO model on a v2 dataset split and report detection metrics."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to a .pt model (default: best.pt)")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="Dataset yaml (default: datasets/v2/vertical.yaml)")
    parser.add_argument("--split", default="test", choices=["val", "test", "test_hard"],
                        help="Split to evaluate (default: test)")
    parser.add_argument("--imgsz", type=int, default=384, help="Long side of the input (default: 384)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 0, cpu, etc. (default: GPU if available, else cpu)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(args.model, args.data, args.split, args.imgsz, args.device)


if __name__ == "__main__":
    main()
