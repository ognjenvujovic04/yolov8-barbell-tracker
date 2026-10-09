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
    resolved = data_dir / 'data.resolved.yaml'
    with open(resolved, 'w') as f:
        yaml.dump(cfg, f, sort_keys=False)
    return str(resolved)

def train_barbell_detector(data_yaml, epochs=30, patience=15, batch_size=8, img_size=320):
    """
    Train YOLOv8n model on barbell dataset at 320×320.
    
    Args:
        data_yaml (str): Path to data.yaml file
        epochs (int): Number of training epochs
        patience (int): Early stopping patience
        batch_size (int): Batch size for training
        img_size (int): Image size for training (320×320)
        
    Returns:
        model: Trained YOLOv8 model
    """
    # Load YOLOv8n model
    model = YOLO('yolov8n.pt')
    
    # Train the model
    print(f"Training YOLOv8n model on barbell dataset...")
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
        name='barbell_detector',
        verbose=True,
        device=device
    )
    
    print(f"Training complete! Model saved in runs/detect/barbell_detector")
    return model

def print_metrics(model):
    """Print final validation metrics from the completed training run."""
    metrics = model.trainer.validator.metrics.box
    print(f"\nFinal Validation Results:")
    print(f"  Precision: {metrics.mp:.4f}")
    print(f"  Recall:    {metrics.mr:.4f}")
    print(f"  mAP@0.5:  {metrics.map50:.4f}")
    print(f"  mAP@0.95: {metrics.map:.4f}")

def main():
    yaml_path = Path(__file__).parent / 'data' / 'data.yaml'
    if not yaml_path.exists():
        exit("Dataset not found!")
    data_yaml = resolve_data_yaml(yaml_path)
    print(f"Using dataset config: {data_yaml}")
    
    img_size = 320
    batch_size = 8
    
    # Train model
    model = train_barbell_detector(
        data_yaml=data_yaml,
        epochs=30,
        patience=8,
        batch_size=batch_size,
        img_size=img_size
    )
    
    print_metrics(model)

    print("\nDone! Weights saved in runs/detect/barbell_detector/weights/")
    
if __name__ == "__main__":
    main()