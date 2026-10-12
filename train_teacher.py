import argparse

import torch
from pathlib import Path
from ultralytics import YOLO

from model import resolve_data_yaml, print_metrics

TEACHER_MODEL = "yolo26l.pt"
FALLBACK_MODEL = "yolo26m.pt"
MIN_VRAM_GB = 4.0


def check_gpu_vram() -> tuple[str, float | None]:
    """Return device string and total VRAM in GB, or None if unavailable."""
    if not torch.cuda.is_available():
        print("Warning: No CUDA GPU found. Training on CPU will be very slow.")
        return "cpu", None

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024**3)
    print(f"GPU: {props.name} ({vram_gb:.1f} GB VRAM)")

    if vram_gb < MIN_VRAM_GB:
        print(
            f"Warning: {vram_gb:.1f} GB VRAM is below the recommended {MIN_VRAM_GB:.0f} GB "
            f"for {TEACHER_MODEL}. If training OOMs, try batch=1 or switch to {FALLBACK_MODEL}."
        )
    return "0", vram_gb


def train_teacher(
    data_yaml: str,
    epochs: int = 100,
    patience: int = 20,
    batch_size: int = 2,
    img_size: int = 640,
    model_name: str = TEACHER_MODEL,
    run_name: str = "barbell_teacher",
) -> YOLO:
    """
    Fine-tune YOLO26l on the barbell dataset at 640x640 for pre-labelling new frames.

    Tuned for RTX 3050 (4 GB VRAM): batch=2, AMP, gradient accumulation (nbs=64).
    The v2 frames are not pre-augmented, so Ultralytics' default on-line augmentation is used.
    model_name can be a previous teacher's best.pt to fine-tune it on more data.
    """
    device, _ = check_gpu_vram()
    if device == "cpu":
        batch_size = 1

    model = YOLO(model_name)

    print(f"Training {model_name} on barbell dataset (teacher model)...")
    print(f"Using data config: {data_yaml}")
    print(f"Image size: {img_size}x{img_size}, batch: {batch_size}, effective batch: 64 (nbs)")

    model.train(
        data=data_yaml,
        imgsz=img_size,
        batch=batch_size,
        nbs=64,
        epochs=epochs,
        patience=patience,
        device=device,
        name=run_name,
        amp=True,
        cache=True,
        workers=2,
        verbose=True,
    )

    print(f"Training complete! Weights saved in {model.trainer.save_dir}")
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the YOLO26l teacher used to pre-label frames.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent / "datasets" / "v2" / "all.yaml",
        help="Dataset yaml (default: datasets/v2/all.yaml)",
    )
    parser.add_argument(
        "--weights",
        default=TEACHER_MODEL,
        help=f"Starting weights: {TEACHER_MODEL} or a previous teacher's best.pt (default: {TEACHER_MODEL})",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100)")
    parser.add_argument("--name", default="barbell_teacher", help="Run name under runs/detect/")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.data.exists():
        raise SystemExit(f"Dataset not found! Expected {args.data}")

    data_yaml = resolve_data_yaml(args.data)
    print(f"Using dataset config: {data_yaml}")

    model = train_teacher(
        data_yaml=data_yaml, epochs=args.epochs, model_name=args.weights, run_name=args.name
    )
    print_metrics(model)


if __name__ == "__main__":
    main()
