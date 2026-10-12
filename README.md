# Barbell Detection & Tracking System

A lightweight YOLO nano barbell detection and tracking system optimized for mobile devices. Two small models (one for vertical, one for horizontal videos) detect and track weightlifting barbells in video footage, with support for path visualization and TFLite export for mobile deployment.

## Features

- **Two Rectangular Models**: 384×224 for vertical and 224×384 for horizontal videos
- **Teacher-Assisted Labelling**: A large YOLO26l pre-labels frames for review in CVAT
- **Leak-Free Evaluation**: Train/val/test split by video, with duplicate and trimmed copies grouped
- **Multi-Barbell Selection**: Interactive selection when multiple barbells are detected
- **Path Visualization**: Draws barbell trajectory with motion trails
- **Mobile-Optimized**: Nano models with a smaller input than the previous 320×320 for low-spec mobile devices
- **TFLite Export**: Convert model to TensorFlow Lite format for mobile deployment
- **Performance Comparison**: Built-in tools to compare PyTorch vs TFLite inference speed

## Demo

**Input Video:**
![Input Video](assets/input_video.gif)

**Output Video with Tracking:**
![Output Video](assets/output_video.gif)

**Barbell Selection**
![Barbell Selection Interface](assets/selection.png)

## Requirements

```bash
pip install -r requirements.txt
```

Training and the dataset scripts are run from the project root in WSL (GPU). Raw videos live in `../data`; everything generated lives in `datasets/` (git-ignored).

## Dataset pipeline (v2)

The app uses two models with rectangular inputs: **vertical 384×224** and **horizontal 224×384** (height×width). A large YOLO26l *teacher* pre-labels new frames, you review them in CVAT, and the nano *students* are trained on the reviewed labels. Train/val/test are split **by video**, so no frames of a test video are ever trained on.

### Labelling rules

CVAT project labels: `barbell`, `ignore`, `check`.

- **barbell**: tight box around the outer plate / bar end of the barbell being lifted, side or diagonal view (up to ~45°), at least 50% visible. Regular Olympic-style barbells only.
- **ignore**: plate-like things that must not count either way: background or racked barbells, stacked plates, a front-view plate. They are painted grey when the dataset is built.
- **No box**: empty bars; frames without a lifted loaded bar are negatives.
- **check**: only appears in label audits; turn every `check` box into `barbell` or `ignore`, or delete it.
- Front/back-view clips, EZ bars, dumbbells and landmine (T-bar row) clips are excluded.

### 1. Video registry, triage and split

```bash
python3 build_index.py --sheets   # datasets/work/videos.csv + contact sheets in datasets/work/triage/
python3 split_dataset.py          # after triage: assign video groups to train/val/test (frozen)
```

`videos.csv` has one row per video: content crop (black bars removed), orientation, `use` (pos / neg / triage / skip / dup / hard) and a duplicate `group` (exact copies and near-duplicates such as trimmed clips). In the triage, set `view=front` for front/back views and resolve `use=triage` to `pos` or `skip`.

### 2. Labelling loop

```bash
python3 extract_frames.py --batch b1_eval --split val test        # 1 fps, max 30 frames per video
python3 prelabel.py --batch b1_eval --weights runs/detect/teacher_t1/weights/best.pt
```

`prelabel.py` writes two CVAT tasks per batch: `check` (frames to review carefully) and `quick` (confident frames to skim). Create a task from each `upload_*` folder, upload the matching zip as **Ultralytics YOLO Detection 1.0**, fix the boxes, and export the task (same format) to `datasets/work/cvat/`. Then rebuild the dataset:

```bash
python3 build_dataset.py   # datasets/v2/{vertical,horizontal}/{train,val,test,test_hard} + vertical/horizontal/all.yaml
```

To find label mistakes, audit labelled frames with the latest teacher and review the flagged frames the same way:

```bash
python3 prelabel.py --audit datasets/v2/all.yaml --weights runs/detect/teacher_t3/weights/best.pt
```

### 3. Teacher (YOLO26l, 640)

```bash
python3 train_teacher.py --data datasets/v2/all.yaml --name teacher_t1
python3 train_teacher.py --data datasets/v2/all.yaml --weights runs/detect/teacher_t1/weights/best.pt --epochs 40 --name teacher_t2
```

Tuned for a 4 GB GPU (batch 2, AMP, gradient accumulation to an effective batch of 64).

### 4. Students (app models)

```bash
python3 model.py --data datasets/v2/all.yaml --weights yolo26n.pt --name student_base_26n
python3 model.py --data datasets/v2/vertical.yaml --weights runs/detect/student_base_26n/weights/best.pt --epochs 40 --name student_vertical
python3 model.py --data datasets/v2/horizontal.yaml --weights runs/detect/student_base_26n/weights/best.pt --epochs 40 --name student_horizontal
```

Students train square at 384 (with mosaic) and are exported with a rectangular input of the same long side, so objects keep the scale they were trained at.

### 5. Evaluate

```bash
python3 test.py --model runs/detect/student_vertical/weights/best.pt --data datasets/v2/vertical.yaml --split val
python3 test.py --model runs/detect/student_vertical/weights/best.pt --data datasets/v2/vertical.yaml --split test_hard
```

Reports Precision, Recall, mAP@0.5, mAP@0.5:0.95 and the confidence with the best F1, which is the app's detection threshold. Use `--split test` only for the final models.

### 6. Export to TFLite

```bash
python3 convert.py --weights runs/detect/student_vertical/weights/best.pt --imgsz 384 224 --name barbell_vertical
python3 convert.py --weights runs/detect/student_horizontal/weights/best.pt --imgsz 224 384 --name barbell_horizontal
python3 test.py --model runs/detect/student_vertical/weights/best.pt --data datasets/v2/vertical.yaml \
    --tflite exports/barbell_vertical_384x224_float32.tflite
```

Writes `exports/barbell_{vertical,horizontal}_{H}x{W}_float{32,16}.tflite` with output `[1, 5, N]` (YOLO26 is exported with `end2end=False`, the YOLOv8 layout the app parses). The last command checks that the TFLite model predicts the same boxes as the `.pt` model.

### 7. Process Video with Tracking

Process a video with barbell detection and path visualization:

```python
python visualize_model.py
```

**Features:**
- **Interactive Selection**: If multiple barbells are detected, click on the one you want to track
- **Motion Trail**: Visualizes barbell path with fading effect (default: 15 frames)
- **Adjustable FPS**: Set target frame rate for output video (default: 25 fps)
- **Real-time Tracking**: Uses YOLO's built-in tracking algorithm

**Configuration in `visualize_model.py`:**
```python
vertical_model = "exports/barbell_vertical_384x224_float32.tflite"      # used when height >= width
horizontal_model = "exports/barbell_horizontal_224x384_float32.tflite"  # used for landscape videos
video_path = "test_video/video.mp4"  # Input video
output_video = "output/output_video.mp4"  # Output path
target_fps = 25                  # Output video FPS
trail_length = 15                # Number of trail dots
```

### 8. Compare Model Performance

Compare inference speed between PyTorch and TFLite versions:

```python
python compare_models.py
```

**Output includes:**
- Detection count
- Bounding box coordinates
- Confidence scores
- Inference time comparison
- Speed ratio between formats

## Model Details

### Training Configuration
- **Students (app)**: YOLO26n or YOLOv8n, trained at 384, exported as 384×224 (vertical) and 224×384 (horizontal)
- **Teacher**: YOLO26l at 640, used only to pre-label and audit frames
- **Dataset**: frames from `../data` videos, labelled in CVAT (the first 2,936 frames come from Roboflow), split by video
- **Classes**: 1 (barbell)
- **Training Device**: Automatic GPU/CPU detection

### Model Performance
The model is optimized for:
- Low-specification mobile devices
- Real-time inference (25+ fps on mobile)
- Accurate barbell detection in various lighting conditions
- Tracking through occlusions

## Data Format

`build_dataset.py` writes YOLO txt labels and data yamls like this (`all.yaml` lists both orientations):

```yaml
path: .
train: vertical/train/images
val: vertical/val/images
test: vertical/test/images
test_hard: vertical/test_hard/images
nc: 1
names: ['barbell']
```

## How It Works

1. **Training** (`model.py`, `train_teacher.py`):
   - Fine-tunes pre-trained YOLO weights on the v2 dataset
   - Validates on held-out videos (export is done with `convert.py`)

2. **Selection** (`visualize_model.py`):
   - Detects all barbells in the first frame
   - Displays interactive selection UI
   - User clicks on desired barbell to track

3. **Tracking** (`visualize_model.py`):
   - Uses YOLO's built-in tracking algorithm
   - Maintains barbell ID across frames
   - Draws bounding box and motion trail

4. **Export** (`convert.py`):
   - Converts PyTorch model to TFLite
   - Optimizes for mobile inference
   - Maintains detection accuracy

## Output Format

The processed video includes:
- **Red bounding box** around tracked barbell
- **Motion trail** showing barbell path (fading effect)
- **Dots** at center points with alpha blending
- **Frame rate** as specified in configuration

## Performance Tips

- **GPU Training**: Automatically uses GPU if available (5-10x faster)
- **Batch Size**: Auto-adjusts for CPU (max 4) to prevent memory issues
- **Image Size**: 384×224 costs about 16% less than a 320×320 input and keeps more detail along the long side
- **Trail Length**: Reduce for better performance, increase for longer visualization

## Troubleshooting

**No barbells detected:**
- Check video quality and lighting
- Ensure barbell is clearly visible in frame
- Try adjusting confidence threshold in code

**Slow processing:**
- Reduce `target_fps` parameter
- Use smaller `trail_length`
- Ensure GPU is being utilized for inference


## Acknowledgments

- **YOLOv8** by Ultralytics for the base detection framework
- **Roboflow** for dataset labeling and management

## Contact

For questions or contributions, please open an issue or submit a pull request.
