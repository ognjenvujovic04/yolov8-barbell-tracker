"""
Export a trained barbell detector to TFLite (float32 + float16) with a rectangular input.

The app's parser expects the YOLOv8-style output [1, 4 + classes, N], so YOLO26 models are
exported with end2end=False (their default TFLite output is NMS-free [1, 300, 6]).

Run from the project root in WSL:
    python3 convert.py --weights runs/detect/student_vertical/weights/best.pt --imgsz 384 224 --name barbell_vertical
    python3 convert.py --weights runs/detect/student_horizontal/weights/best.pt --imgsz 224 384 --name barbell_horizontal
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
EXPORT_DIR = ROOT / "exports"
STRIDES = (8, 16, 32)


def expected_anchors(h: int, w: int) -> int:
    return sum((h // s) * (w // s) for s in STRIDES)


def check_tflite(path: Path, h: int, w: int) -> None:
    """Assert the TFLite input is [1, h, w, 3] and the output is [1, 5, N] for one class."""
    interpreter = tf.lite.Interpreter(model_path=str(path))
    in_shape = list(interpreter.get_input_details()[0]["shape"])
    out_shape = list(interpreter.get_output_details()[0]["shape"])
    assert in_shape == [1, h, w, 3], f"{path.name}: input {in_shape}, expected [1, {h}, {w}, 3]"
    assert out_shape == [1, 5, expected_anchors(h, w)], (
        f"{path.name}: output {out_shape}, expected [1, 5, {expected_anchors(h, w)}]"
    )
    print(f"{path.name}: input {in_shape}, output {out_shape}")


def export(weights: Path, h: int, w: int, name: str, image: Path | None) -> list[Path]:
    model = YOLO(str(weights))
    saved_model = Path(model.export(format="tflite", imgsz=[h, w], end2end=False)).parent

    EXPORT_DIR.mkdir(exist_ok=True)
    outputs = []
    for precision in ("float32", "float16"):
        src = next(saved_model.glob(f"*_{precision}.tflite"))
        dst = EXPORT_DIR / f"{name}_{h}x{w}_{precision}.tflite"
        shutil.copy2(src, dst)
        check_tflite(dst, h, w)
        outputs.append(dst)

    if image:
        results = YOLO(str(outputs[0]), task="detect")(str(image), imgsz=[h, w])
        for box in results[0].boxes:
            print(f"  {image.name}: box {np.round(box.xyxy[0].tolist(), 1)} conf {float(box.conf[0]):.3f}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a barbell detector to rectangular TFLite models.")
    parser.add_argument("--weights", type=Path, required=True, help="Trained weights (.pt)")
    parser.add_argument("--imgsz", type=int, nargs=2, required=True, metavar=("H", "W"),
                        help="Input height and width, multiples of 32 (vertical: 384 224, horizontal: 224 384)")
    parser.add_argument("--name", required=True, help="Output name, e.g. barbell_vertical")
    parser.add_argument("--image", type=Path, help="Optional image to run the float32 model on")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    h, w = args.imgsz
    if h % 32 or w % 32:
        raise SystemExit("--imgsz values must be multiples of 32")
    for path in export(args.weights, h, w, args.name, args.image):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
