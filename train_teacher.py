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
    epochs: int = 30,
    patience: int = 15,
    batch_size: int = 2,
    img_size: int = 640,
    model_name: str = TEACHER_MODEL,
) -> YOLO:
    """
    Fine-tune YOLO26l on the barbell dataset at 640x640 for pseudo-label generation.

    Tuned for RTX 3050 (4 GB VRAM): batch=2, AMP, gradient accumulation (nbs=64).
    Light on-line augmentation (mosaic only); images are already augmented in Roboflow.
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
        name="barbell_teacher",
        amp=True,
        cache=True,
        workers=2,
        verbose=True,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
    )

    print("Training complete! Weights saved in runs/detect/barbell_teacher/")
    return model


def main():
    yaml_path = Path(__file__).parent / "data" / "data.yaml"
    if not yaml_path.exists():
        raise SystemExit("Dataset not found! Expected data/data.yaml")

    data_yaml = resolve_data_yaml(yaml_path)
    print(f"Using dataset config: {data_yaml}")

    model = train_teacher(data_yaml=data_yaml)
    print_metrics(model)


if __name__ == "__main__":
    main()
