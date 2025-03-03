import cv2
import numpy as np
import time
import datetime
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
    first_results = model.track(first_frame_rgb, persist=True)[0] 
    
    # Create a list of detected barbells
    barbell_boxes = []
    for box in first_results.boxes:
        if int(box.cls[0]) == 0: 
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else None  # Get the tracking ID
            barbell_boxes.append({
                'id': track_id, 
                'box': (x1, y1, x2, y2),
                'center': (center_x, center_y),
                'conf': conf
            })
    
    # If no barbells detected
    if not barbell_boxes:
        print("No barbells detected in the first frame.")
        return
    
    # If only one barbell detected
    if len(barbell_boxes) == 1:
        print("Only one barbell detected.")
        return barbell_boxes[0]
    
    # Draw detected barbells on the frame
    for bb in barbell_boxes:
        x1, y1, x2, y2 = bb['box']
        cv2.rectangle(first_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(first_frame, f"ID: {bb['id']} Conf: {bb['conf']:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Function to handle mouse clicks
    selected_barbell = None
    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_barbell
        if event == cv2.EVENT_LBUTTONDOWN:
            for bb in barbell_boxes:
                x1, y1, x2, y2 = bb['box']
                if x1 <= x <= x2 and y1 <= y <= y2:
                    selected_barbell = bb
                    break
    
    # Display the frame and wait for user selection
    cv2.imshow("Select a Barbell", first_frame)
    cv2.setMouseCallback("Select a Barbell", mouse_callback)
    
    print("Click on the barbell you want to track.")
    while selected_barbell is None:
        cv2.waitKey(1)
    
    cv2.destroyAllWindows()
    
    print(f"Selected barbell ID: {selected_barbell['id']} with confidence {selected_barbell['conf']:.2f}")
    return selected_barbell

if __name__ == "__main__":
    model_path = "best.pt"
    video_path = "test_video/video.mp4"
    output_video = "output/output_video.mp4"
    start_time = time.time()
    process_video(model_path, video_path, output_video, 15, 40)
    end_time = time.time()
    print(f"Processing time: {end_time - start_time} seconds")
