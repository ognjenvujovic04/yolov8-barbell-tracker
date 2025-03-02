import os
import yaml
import torch
from ultralytics import YOLO
from pathlib import Path

def train_barbell_detector(data_yaml, epochs=50, patience=15, batch_size=8, img_size=768):
    """
    Train YOLOv8n model on barbell dataset with 720×900 frames
    
    Args:
        data_yaml (str): Path to data.yaml file
        epochs (int): Number of training epochs
        patience (int): Early stopping patience
        batch_size (int): Batch size for training
        img_size (int): Image size for training (will be resized to square)
        
    Returns:
        model: Trained YOLOv8 model
    """
    # Load YOLOv8n model
    model = YOLO('yolov8n.pt')
    
    # Train the model
    print(f"Training YOLOv8n model on barbell dataset...")
    print(f"Using data config: {data_yaml}")
    print(f"Original image size is 720×900, using {img_size}×{img_size} for training")
    
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
        imgsz=img_size,  # Resized to square
        name='barbell_detector',
        verbose=True,
        device=device
    )
    
    print(f"Training complete! Model saved in runs/detect/barbell_detector")
    return model

def validate_model(model, data_yaml, img_size=768):
    """
    Validate the trained model
    
    Args:
        model: Trained YOLOv8 model
        data_yaml: Path to data.yaml file
        img_size: Image size for validation
    """
    # Run validation
    print("Validating model...")
    results = model.val(data=data_yaml, imgsz=img_size)
    
    # Print metrics
    precision = results.box.maps[0]  # mAP at 0.5 IoU
    recall = results.box.mar[0]      # mAR at 0.5 IoU
    
    print(f"Validation Results:")
    print(f"mAP@0.5: {precision:.4f}")
    print(f"mAR@0.5: {recall:.4f}")

def main():
    # Check if YAML file already exists
    yaml_path = Path('data/data.yaml')
    if yaml_path.exists():
        print(f"Using existing data.yaml file at {yaml_path}")
        data_yaml = str(yaml_path)
    else:
        exit("Dataset not found!")
    
    # Choose image size based on original 720×900
    # Using 768 as it's a multiple of 32 (YOLOv8 works best with sizes divisible by 32)
    img_size = 768
    
    # Adjust batch size based on memory constraints
    # Smaller batch size for larger images to avoid out-of-memory errors
    batch_size = 8
    
    # Train model
    model = train_barbell_detector(
        data_yaml=data_yaml,
        epochs=50,
        patience=15,
        batch_size=batch_size,
        img_size=img_size
    )
    
    # Validate the model
    validate_model(model, data_yaml, img_size=img_size)
    
    # Save the model
    model_path = 'barbell_detector.pt'
    model.export(format='pt', save=True)
    print(f"Model exported to {model_path}")
    
    print("Done! You can now use the trained model for barbell detection.")
    
if __name__ == "__main__":
    main()