import cv2
import numpy as np
import time
import datetime
import os
from PIL import Image, ImageDraw
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
    
    selected_barbell = select_barbell_detections(cap, model)
    if not selected_barbell:
        print("Error: No barbell selected.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC)  
        
        if current_time - last_frame_time >= frame_time:  
            last_frame_time = current_time
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(image)[0]

            # Convert to PIL Image for better transparency support
            pil_image = Image.fromarray((image * 0.5).astype(np.uint8))
            
            overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Compute center of bounding box
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                previous_dots.append((center_x, center_y))  

                # Keep only the last 'trail_length' dots
                if len(previous_dots) > trail_length:
                    previous_dots.pop(0)
            
            # Draw the bounding boxes on the main image
            pil_draw = ImageDraw.Draw(pil_image)
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pil_draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=2)

            # Draw previous dots and connect them with lines on the overlay
            for i in range(1, len(previous_dots)):
                # Calculate transparency (0-255)
                alpha = int(255 * ((i / len(previous_dots))))
                
                # Draw line with transparency
                draw.line([previous_dots[i-1], previous_dots[i]], fill=(255, 0, 0, alpha), width=13)
                
                # Draw circle with transparency
                x, y = previous_dots[i]
                draw.ellipse((x-6, y-6, x+6, y+6), fill=(255, 0, 0, alpha))
            
            pil_image = pil_image.convert('RGBA')
            composited = Image.alpha_composite(pil_image, overlay)
            
            final_image = np.array(composited.convert('RGB'))
            
            # Convert back to BGR for saving
            output_frame = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
            out.write(output_frame)

    cap.release()
    out.release()
    print(f"Output video saved as {output_video}")
    
def select_barbell_detections(cap, model):
    # Get first frame for barbell selection
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return
    
    # Reset the video capture to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Get detections for first frame
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    first_results = model(first_frame_rgb)[0]
    
    # Create a list of detected barbells
    barbell_boxes = []
    for i, box in enumerate(first_results.boxes):
        if int(box.cls[0]) == 0:  # Assuming barbell is class 0, adjust if needed
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            conf = float(box.conf[0])
            barbell_boxes.append({
                'id': i,
                'box': (x1, y1, x2, y2),
                'center': (center_x, center_y),
                'conf': conf
            })
    
    # If no barbells detected
    if not barbell_boxes:
        print("No barbells detected in the first frame.")
        return
    
    # User selection of barbell
    print("\nDetected barbells in first frame:")
    for i, bb in enumerate(barbell_boxes):
        print(f"{i+1}. Barbell at position {bb['center']} with confidence {bb['conf']:.2f}")
    
    selection = -1
    while selection < 0 or selection >= len(barbell_boxes):
        try:
            selection = int(input(f"Select a barbell to track (1-{len(barbell_boxes)}): ")) - 1
        except ValueError:
            print("Please enter a valid number.")
            
    selected_barbell = barbell_boxes[selection]
    print(f"Selected barbell: {selected_barbell['center']} with confidence {selected_barbell['conf']:.2f}")
    return selected_barbell
    

if __name__ == "__main__":
    model_path = "best.pt"
    video_path = "test_video/video.mp4"
    output_video = "output/output_video.mp4"
    start_time = time.time()
    process_video(model_path, video_path, output_video, 15, 40)
    end_time = time.time()
    print(f"Processing time: {end_time - start_time} seconds")
