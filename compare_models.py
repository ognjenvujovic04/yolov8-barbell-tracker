import cv2
import numpy as np
import time
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image

# Paths
pytorch_model_path = "best.pt"
tflite_model_path = "best_saved_model/best_float32.tflite"
image_path = "test_frames/deadlift_g7.jpg"

def load_image(image_path):
    """Load and preprocess image for both models"""
    # Read image for PyTorch model (using cv2)
    img_cv = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    # Read image for TFLite model (using PIL)
    img_pil = Image.open(image_path)
    
    return img_rgb, img_pil, img_cv.shape[:2]  # Return height, width

def run_pytorch_inference(model_path, img):
    """Run inference with PyTorch YOLOv8 model"""
    # Load the model
    model = YOLO(model_path)
    
    # Start timing
    start_time = time.time()
    
    # Run inference
    results = model(img)
    
    # End timing
    inference_time = time.time() - start_time
    
    return results[0], inference_time  # Return the first result and inference time

def run_tflite_inference(model_path, img_pil, input_shape=(768, 768)):
    """Run inference with TFLite YOLOv8 model"""
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Preprocess image (resize to the expected input shape)
    img_resized = img_pil.resize(input_shape)
    input_data = np.array(img_resized, dtype=np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Start timing
    start_time = time.time()
    
    # Run inference
    interpreter.invoke()
    
    # End timing
    inference_time = time.time() - start_time
    
    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    return output_data, inference_time

def process_tflite_output(output_data, img_shape, conf_threshold=0.025):
    """Process TFLite output to extract bounding boxes and confidence values"""
    # Get image dimensions
    img_height, img_width = img_shape
    
    detections = []
    
    # Process each detection
    for detection in output_data[0]:
        # Extract confidence score
        confidence = detection[4]
        
        if confidence >= conf_threshold:
            # Extract class probabilities and find the class with highest probability
            class_scores = detection[5:]
            class_id = np.argmax(class_scores)
            class_score = class_scores[class_id]
            
            # Combined confidence
            final_confidence = float(confidence * class_score)
            
            # Only proceed if the class probability meets the threshold
            if class_score >= conf_threshold:
                # Extract bounding box coordinates (normalized)
                x, y, w, h = detection[0:4]
                
                # Convert to pixel coordinates
                x1 = int((x - w/2) * img_width)
                y1 = int((y - h/2) * img_height)
                x2 = int((x + w/2) * img_width)
                y2 = int((y + h/2) * img_height)
                
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": final_confidence,
                    "class_id": int(class_id)
                })
    
    return detections

def main():
    # Load image
    img_rgb, img_pil, img_shape = load_image(image_path)
    
    print(f"Image loaded: {image_path}")
    print(f"Image shape: {img_shape}")
    
    # Run PyTorch inference
    print("\n=== PyTorch YOLOv8 Results ===")
    pytorch_results, pytorch_time = run_pytorch_inference(pytorch_model_path, img_rgb)
    
    # Print PyTorch results
    print(f"Inference time: {pytorch_time:.4f} seconds")
    print(f"Number of detections: {len(pytorch_results.boxes)}")
    print("\nBounding Boxes:")
    
    for i, box in enumerate(pytorch_results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        
        print(f"Detection {i+1}:")
        print(f"  Confidence: {conf:.4f}")
        print(f"  Bounding Box: [x1={x1}, y1={y1}, x2={x2}, y2={y2}]")
    
    # Run TFLite inference
    print("\n=== TFLite YOLOv8 Results ===")
    tflite_output, tflite_time = run_tflite_inference(tflite_model_path, img_pil)
    
    # Process TFLite output
    tflite_detections = process_tflite_output(tflite_output, img_shape)
    
    # Print TFLite results
    print(f"Inference time: {tflite_time:.4f} seconds")
    print(f"Number of detections: {len(tflite_detections)}")
    print("\nBounding Boxes:")
    
    for i, detection in enumerate(tflite_detections):
        x1, y1, x2, y2 = detection["bbox"]
        conf = detection["confidence"]
        
        print(f"Detection {i+1}:")
        print(f"  Confidence: {conf:.4f}")
        print(f"  Bounding Box: [x1={x1}, y1={y1}, x2={x2}, y2={y2}]")
    
    # Print speed comparison
    print("\n=== Performance Comparison ===")
    print(f"PyTorch inference time: {pytorch_time:.4f} seconds")
    print(f"TFLite inference time: {tflite_time:.4f} seconds")
    print(f"Speed ratio: {(pytorch_time / tflite_time if tflite_time > 0 else 0):.2f}x")

if __name__ == "__main__":
    main()