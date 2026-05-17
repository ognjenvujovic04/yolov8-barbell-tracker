"""
Evaluate the barbell detector on the held-out test set (data/test).

Run from the project root in WSL:
    python test.py
    python test.py --model best.pt
    python test.py --model best_saved_model/best_float32.tflite
"""

import argparse
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_MODEL = ROOT / "best.pt"


def build_data_yaml() -> Path:
    """Build a data.yaml with paths relative to this repo (works in WSL/Linux)."""
    for split in ("train", "valid", "test"):
        images = DATA_DIR / split / "images"
        labels = DATA_DIR / split / "labels"
        if not images.is_dir():
            raise FileNotFoundError(f"Missing dataset folder: {images}")
        if split == "test" and not labels.is_dir():
            raise FileNotFoundError(f"Missing labels folder: {labels}")

    config = {
        "path": str(DATA_DIR),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": ["barbell"],
    }

    out = DATA_DIR / "test_eval.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return out


def count_test_images() -> int:
    images_dir = DATA_DIR / "test" / "images"
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for p in images_dir.iterdir() if p.suffix.lower() in exts)


def get_model_imgsz(model: YOLO) -> int:
    """Read inference size baked into the model checkpoint/export."""
    if hasattr(model.model, "args") and hasattr(model.model.args, "imgsz"):
        imgsz = model.model.args.imgsz
    elif "imgsz" in model.overrides:
        imgsz = model.overrides["imgsz"]
    else:
        return 0
    return int(imgsz[0] if isinstance(imgsz, (list, tuple)) else imgsz)


def print_metrics(metrics, split: str, model_path: Path, imgsz: int, n_images: int) -> None:
    box = metrics.box
    print("\n" + "=" * 60)
    print(f"Barbell detector — {split} set evaluation")
    print("=" * 60)
    print(f"Model:      {model_path}")
    if imgsz:
        print(f"Image size: {imgsz}x{imgsz}")
    print(f"Images:     {n_images}")
    print("-" * 60)
    print(f"Precision (P):     {box.mp:.4f}")
    print(f"Recall (R):        {box.mr:.4f}")
    print(f"mAP@0.5:           {box.map50:.4f}")
    print(f"mAP@0.5:0.95:      {box.map:.4f}")
    if hasattr(box, "maps") and len(box.maps):
        print(f"Per-class mAP@0.5: {box.maps[0]:.4f}")
    print("=" * 60)
    if metrics.speed:
        print(
            f"Inference: {metrics.speed.get('inference', 0):.1f} ms/img  "
            f"(preprocess {metrics.speed.get('preprocess', 0):.1f}, "
            f"postprocess {metrics.speed.get('postprocess', 0):.1f})"
        )
    print(f"Results saved to: {metrics.save_dir}")


def evaluate(model_path: Path, device: str | None) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    data_yaml = build_data_yaml()
    n_test = count_test_images()
    if n_test == 0:
        raise RuntimeError("No images found in data/test/images")

    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {model_path}")
    print(f"Device: {device}")
    print(f"Test images: {n_test}")
    print(f"Data config: {data_yaml}")

    model = YOLO(str(model_path))
    imgsz = get_model_imgsz(model)
    if imgsz:
        print(f"Model input size: {imgsz}x{imgsz}")

    metrics = model.val(
        data=str(data_yaml),
        split="test",
        device=device,
        verbose=True,
        plots=True,
        save_json=True,
    )

    print_metrics(metrics, "test", model_path, imgsz, n_test)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate barbell YOLO model on data/test and report detection metrics."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to .pt or exported TFLite model (default: best.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: 0, cpu, etc. (default: GPU if available, else cpu)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(args.model, args.device)


if __name__ == "__main__":
    main()
