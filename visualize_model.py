import cv2
import numpy as np
import time
import os
from ultralytics import YOLO

def process_video(model_path, video_path, output_video, target_fps=15):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    else:
        print("Video loaded successfully.")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Original FPS: {original_fps}")
    print(f"Target FPS: {target_fps}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video resolution: {width}x{height}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter(output_video, fourcc, target_fps, (width, height))

    frame_time = 1000 / target_fps
    last_frame_time = 0
    previous_dots = []

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
                
                # Draw bounding box
                cv2.rectangle(dimmed_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Compute center of bounding box
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                previous_dots.append((center_x, center_y))  # Store dot position

            # Draw previous dots
            for dot in previous_dots:
                cv2.circle(dimmed_image, dot, radius=6, color=(255, 0, 0), thickness=-1)

            # Convert back to BGR for saving
            output_frame = cv2.cvtColor(dimmed_image, cv2.COLOR_RGB2BGR)
            out.write(output_frame)

    cap.release()
    out.release()
    print(f"Output video saved as {output_video}")

if __name__ == "__main__":
    model_path = "best.pt"
    video_path = "test_video/video.mp4"
    output_video = "output/output_video.mp4"
    start_time = time.time()
    process_video(model_path, video_path, output_video, target_fps=15)
    end_time = time.time()
    print(f"Processing time: {end_time - start_time} seconds")
