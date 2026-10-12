import cv2
import numpy as np
import re
import time
import datetime
from pathlib import Path
from PIL import Image, ImageDraw
from ultralytics import YOLO

def pick_model(video_path, vertical_model, horizontal_model):
    """Choose the vertical or horizontal model (and its input size) from the video's shape."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    model_path = vertical_model if height >= width else horizontal_model
    # Exported models have a fixed input, named like barbell_vertical_384x224_float32.tflite
    size = re.search(r"_(\d+)x(\d+)_", Path(model_path).name)
    imgsz = [int(size.group(1)), int(size.group(2))] if size else 384
    return model_path, imgsz

def process_video(model_path, video_path, output_video, target_fps=15, trail_length=20, imgsz=384):
    model = YOLO(model_path, task="detect")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    print(f"Model: {model_path} (input {imgsz})")

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

    selected_barbell = select_barbell_detections(cap, model, imgsz)
    if not selected_barbell:
        print("Error: No barbell selected.")
        return
    
    selected_id = selected_barbell['id']  
    print(f"Tracking barbell ID: {selected_id}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC)  
        
        if current_time - last_frame_time >= frame_time:  
            last_frame_time = current_time
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.track(image, persist=True, imgsz=imgsz)[0]

            # Convert to PIL Image for better transparency support
            pil_image = Image.fromarray((image * 0.5).astype(np.uint8))
            overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            for box in results.boxes:
                track_id = int(box.id[0]) if box.id is not None else None
                
                # Filter to only track the selected barbell
                if track_id == selected_id:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    previous_dots.append((center_x, center_y))  

                    # Keep only the last 'trail_length' dots
                    if len(previous_dots) > trail_length:
                        previous_dots.pop(0)
            
            # Draw the bounding boxes on the main image
            pil_draw = ImageDraw.Draw(pil_image)
            for box in results.boxes:
                track_id = int(box.id[0]) if box.id is not None else None
                if track_id == selected_id:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    pil_draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=2)

            # Draw previous dots and connect them with lines on the overlay
            for i in range(1, len(previous_dots)):
                alpha = int(255 * ((i / len(previous_dots))))
                draw.line([previous_dots[i-1], previous_dots[i]], fill=(255, 0, 0, alpha), width=13)
                x, y = previous_dots[i]
                draw.ellipse((x-6, y-6, x+6, y+6), fill=(255, 0, 0, alpha))
            
            pil_image = pil_image.convert('RGBA')
            composited = Image.alpha_composite(pil_image, overlay)
            final_image = np.array(composited.convert('RGB'))
            
            output_frame = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
            out.write(output_frame)

    cap.release()
    out.release()
    print(f"Output video saved as {output_video}")

def select_barbell_detections(cap, model, imgsz=384):
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    first_results = model.track(first_frame_rgb, persist=True, imgsz=imgsz)[0]

    barbell_boxes = []
    for box in first_results.boxes:
        track_id = int(box.id[0]) if box.id is not None else None
        if track_id is not None:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            conf = float(box.conf[0])
            barbell_boxes.append({
                'id': track_id, 
                'box': (x1, y1, x2, y2),
                'center': (center_x, center_y),
                'conf': conf
            })
    
    if not barbell_boxes:
        print("No barbells detected in the first frame.")
        return
    
    if len(barbell_boxes) == 1:
        print("Only one barbell detected.")
        return barbell_boxes[0]
    
    for bb in barbell_boxes:
        x1, y1, x2, y2 = bb['box']
        cv2.rectangle(first_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(first_frame, f"ID: {bb['id']} Conf: {bb['conf']:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    selected_barbell = None
    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_barbell
        if event == cv2.EVENT_LBUTTONDOWN:
            for bb in barbell_boxes:
                x1, y1, x2, y2 = bb['box']
                if x1 <= x <= x2 and y1 <= y <= y2:
                    selected_barbell = bb
                    break
    
    cv2.imshow("Select a Barbell", first_frame)
    cv2.setMouseCallback("Select a Barbell", mouse_callback)
    
    print("Click on the barbell you want to track.")
    while selected_barbell is None:
        cv2.waitKey(1)
    
    cv2.destroyAllWindows()
    
    print(f"Selected barbell ID: {selected_barbell['id']} with confidence {selected_barbell['conf']:.2f}")
    return selected_barbell

if __name__ == "__main__":
    vertical_model = "exports/barbell_vertical_384x224_float32.tflite"
    horizontal_model = "exports/barbell_horizontal_224x384_float32.tflite"
    video_path = "test_video/video.mp4"
    output_video = "output/output_video.mp4"
    start_time = time.time()
    model_path, imgsz = pick_model(video_path, vertical_model, horizontal_model)
    process_video(model_path, video_path, output_video, 25, 15, imgsz)
    end_time = time.time()
    print(f"Processing time: {end_time - start_time} seconds")
