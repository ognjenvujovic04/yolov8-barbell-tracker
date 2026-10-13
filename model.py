import argparse

import albumentations as A
import yaml
import torch
from ultralytics import YOLO
from pathlib import Path

def resolve_data_yaml(yaml_path: Path) -> str:
    """Write a data yaml with absolute dataset root (avoids Ultralytics datasets_dir cache)."""
    data_dir = yaml_path.parent.resolve()
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    cfg['path'] = str(data_dir)
    resolved = data_dir / f'{yaml_path.stem}.resolved.yaml'
    with open(resolved, 'w') as f:
        yaml.dump(cfg, f, sort_keys=False)
    return str(resolved)

def domain_augmentations():
    """
    Pixel-level augmentations for phone videos of lifts, on top of Ultralytics' defaults
    (mosaic, HSV, scale, translate, fliplr). Replaces Ultralytics' Albumentations list
    (Blur, MedianBlur, ToGray, CLAHE at p=0.01).

    Rotation and shear stay off (degrees=0, shear=0): the axis-aligned box of a rotated round
    plate is larger than the plate, and the app derives cm/pixel from the box size.
    """
    return [
        A.MotionBlur(blur_limit=(3, 9), p=0.2),             # fast bar in pulls and jerks
        A.ImageCompression(quality_range=(40, 90), p=0.2),  # re-encoded phone / TikTok video
        A.GaussNoise(std_range=(0.02, 0.06), p=0.1),        # sensor noise in dim gyms
    ]

def train_barbell_detector(data_yaml, weights='yolo26n.pt', epochs=100, patience=20, batch_size=16,
                           img_size=384, run_name='barbell_detector', domain_aug=True):
    """
    Train the nano barbell detector (student model used in the app).

    Trains square (with mosaic) at img_size. The model is later exported with a rectangular
    input of the same long side (e.g. 384×224), so objects keep the scale seen in training.
    Augmentation is on-line (redrawn every epoch); no augmented copies are written to the dataset.

    Args:
        data_yaml (str): Path to data.yaml file
        weights (str): Starting weights: yolo26n.pt, yolov8n.pt or a base run's best.pt to fine-tune
        epochs (int): Number of training epochs
        patience (int): Early stopping patience
        batch_size (int): Batch size for training
        img_size (int): Training image size (long side of the exported input)
        run_name (str): Run name under runs/detect/
        domain_aug (bool): Add domain_augmentations(); False keeps Ultralytics' defaults only

    Returns:
        model: Trained YOLO model
    """
    model = YOLO(weights)

    print(f"Training {weights} on barbell dataset...")
    print(f"Using data config: {data_yaml}")
    print(f"Training at {img_size}×{img_size}")
    
    # Check for GPU
    device = '0' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("Warning: Training on CPU. This may be slow. Consider using a GPU if available.")
        # Reduce batch size on CPU
        batch_size = min(4, batch_size)
    else:
        print("GPU is avaliable")
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        patience=patience,
        batch=batch_size,
        imgsz=img_size,
        name=run_name,
        verbose=True,
        device=device,
        augmentations=domain_augmentations() if domain_aug else None,
    )

    print(f"Training complete! Model saved in {model.trainer.save_dir}")
    return model

def print_metrics(model):
    """Print final validation metrics from the completed training run."""
    metrics = model.trainer.validator.metrics.box
    print(f"\nFinal Validation Results:")
    print(f"  Precision: {metrics.mp:.4f}")
    print(f"  Recall:    {metrics.mr:.4f}")
    print(f"  mAP@0.5:  {metrics.map50:.4f}")
    print(f"  mAP@0.95: {metrics.map:.4f}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train the nano barbell detector used in the app.")
    parser.add_argument('--data', type=Path, default=Path(__file__).parent / 'datasets' / 'v2' / 'all.yaml',
                        help="Dataset yaml (default: datasets/v2/all.yaml)")
    parser.add_argument('--weights', default='yolo26n.pt',
                        help="Starting weights: yolo26n.pt, yolov8n.pt or a base run's best.pt (default: yolo26n.pt)")
    parser.add_argument('--imgsz', type=int, default=384, help="Training image size (default: 384)")
    parser.add_argument('--epochs', type=int, default=100, help="Training epochs (default: 100)")
    parser.add_argument('--patience', type=int, default=20, help="Early stopping patience (default: 20)")
    parser.add_argument('--batch', type=int, default=16, help="Batch size (default: 16, at most 4 on CPU)")
    parser.add_argument('--name', default='barbell_detector', help="Run name under runs/detect/")
    parser.add_argument('--no-domain-aug', action='store_true',
                        help="Train without the motion blur / compression / noise augmentations")
    return parser.parse_args()

def main():
    args = parse_args()
    if not args.data.exists():
        exit(f"Dataset not found: {args.data}")
    data_yaml = resolve_data_yaml(args.data)
    print(f"Using dataset config: {data_yaml}")

    model = train_barbell_detector(
        data_yaml=data_yaml,
        weights=args.weights,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch,
        img_size=args.imgsz,
        run_name=args.name,
        domain_aug=not args.no_domain_aug,
    )

    print_metrics(model)

if __name__ == "__main__":
    main()