import cv2
import numpy as np
import time
import datetime
import os
from ultralytics import YOLO

def process_video(model_path, video_path, output_video, target_fps=15, trail_length=20):
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
    
        
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) 
    video_time = datetime.timedelta(seconds=(frames / original_fps))
    print(f"Video duration: {video_time}")

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
                previous_dots.append((center_x, center_y))  

                # Keep only the last 'trail_length' dots
                if len(previous_dots) > trail_length:
                    previous_dots.pop(0)

            # Draw previous dots and connect them with lines
            for i in range(1, len(previous_dots)):
                alpha = 1 - (i / len(previous_dots)) 
                faded_color = (int(255 * (1 - alpha)), 0, 0)

                cv2.line(dimmed_image, previous_dots[i - 1], previous_dots[i], faded_color, 12)
                cv2.circle(dimmed_image, previous_dots[i], radius=6, color=faded_color, thickness=-1)

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
    process_video(model_path, video_path, output_video, 15, 30)
    end_time = time.time()
    print(f"Processing time: {end_time - start_time} seconds")
