import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

def visualize_results(model_path, frames_folder, output_folder):
    model = YOLO(model_path)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    frame_paths = [os.path.join(frames_folder, f) for f in os.listdir(frames_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for frame_path in frame_paths:
        image = cv2.imread(frame_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        
        results = model(image)[0]
        
        dimmed_image = (image * 0.5).astype(np.uint8)
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(dimmed_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Red bounding box
        
        output_path = os.path.join(output_folder, os.path.basename(frame_path))
        cv2.imwrite(output_path, cv2.cvtColor(dimmed_image, cv2.COLOR_RGB2BGR))
        
        plt.imshow(dimmed_image)
        plt.axis("off")
        plt.show(block=False)
        plt.pause(2) 
        plt.close()

if __name__ == "__main__":
    model_path = "best.pt"  
    frames_folder = "test_frames"
    output_folder = "output_frames"
    visualize_results(model_path, frames_folder, output_folder)
