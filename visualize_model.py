import cv2
import numpy as np
from ultralytics import YOLO
import os

def process_video(model_path, video_path, output_folder, target_fps=15):
    model = YOLO(model_path)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'")
        return
    else:
        print(f"Processing video '{video_path}'")
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Original FPS: {original_fps}")
    print(f"Target FPS: {target_fps}")
    frame_time = 1000 / target_fps 
    last_frame_time = 0  
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) 
        
        if current_time - last_frame_time >= frame_time:  
            last_frame_time = current_time
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(image)[0]

            dimmed_image = (image * 0.5).astype(np.uint8)

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(dimmed_image, (x1, y1), (x2, y2), (255, 0, 0), 2) 
            
            output_path = os.path.join(output_folder, f"frame_{int(current_time)}.jpg")
            cv2.imwrite(output_path, cv2.cvtColor(dimmed_image, cv2.COLOR_RGB2BGR))

    cap.release()

if __name__ == "__main__":
    model_path = "best.pt"
    video_path = "test_video/video.mp4"
    output_folder = "output_frames"
    process_video(model_path, video_path, output_folder, target_fps=10)
