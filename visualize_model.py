import cv2
import numpy as np
from ultralytics import YOLO
import os

def visualize_results(model_path, video_path, output_folder):
    model = YOLO(model_path)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(image)[0]
        
        dimmed_image = (image * 0.5).astype(np.uint8)
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(dimmed_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Red bounding box
        
        output_path = os.path.join(output_folder, f"frame_{frame_idx:04d}.png")
        cv2.imwrite(output_path, cv2.cvtColor(dimmed_image, cv2.COLOR_RGB2BGR))
                
        frame_idx += 1
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = "best.pt"  
    video_path = "test_video/video.mp4"
    output_folder = "output_frames"
    visualize_results(model_path, video_path, output_folder)
