from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("best.pt")

# Export the model to TFLite format
model.export(format="tflite", int8=True, imgsz=640)

# Load the exported TFLite model
tflite_model = YOLO("best_saved_model/best_float32.tflite")

# Run inference
results = tflite_model("test_frames/deadlift_g7.jpg")


# Print results
for box in results[0].boxes:
    print(box.xyxy, box.conf)