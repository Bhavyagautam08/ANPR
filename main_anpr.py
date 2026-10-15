import os
import cv2
import logging
from pathlib import Path
from ultralytics import YOLO

# Replace logging configuration with this:
logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("paddleseg").setLevel(logging.ERROR)
logging.getLogger("paddle").setLevel(logging.ERROR)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# -------------------- CONFIGURATION --------------------
ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT / "runs" / "plate_detection" / "yolo11n_plate_run" / "weights" / "best.pt"
TEST_IMAGE = ROOT / "Dataset_1" / "State-wise_OLX" / "AN" / "AN1.jpg"
OUTPUT_DIR = ROOT / "output_results"

def clean_text(text):
    """Filter out non-alphanumeric noise for strict license plate strings."""
    return "".join(ch for ch in text.upper() if ch.isalnum())

def run_anpr(image_path):
    if not image_path.exists():
        print(f"Error: Image does not exist at {image_path}")
        return

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    h, w = img.shape[:2]

    # 1. Run YOLO Plate Detection on GPU
    print("Loading YOLO11 plate detector (GPU)...")
    yolo_model = YOLO(str(WEIGHTS_PATH))
    results = yolo_model(img, conf=0.5)[0]
    boxes = results.boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        print("No license plates detected.")
        return

    print(f"Detected {len(boxes)} license plate(s).")

    # Crop detected plates
    crops = []
    box_coords = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        # Add 5% padding around crop
        mx, my = int((x2 - x1) * 0.05), int((y2 - y1) * 0.05)
        cx1, cy1 = max(0, x1 - mx), max(0, y1 - my)
        cx2, cy2 = min(w, x2 + mx), min(h, y2 + my)

        crops.append(img[cy1:cy2, cx1:cx2])
        box_coords.append((x1, y1, x2, y2))

    # Free YOLO model memory before loading PaddleOCR
    del yolo_model

    # 2. Lazy-import PaddleOCR and run text recognition (CPU mode avoids CUDA clash)
    print("Loading PaddleOCR engine...")
    from paddleocr import PaddleOCR
    ocr_model = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

    for i, (plate_crop, (x1, y1, x2, y2)) in enumerate(zip(crops, box_coords), 1):
        ocr_res = ocr_model.ocr(plate_crop, cls=True)
        detected_text = ""
        
        if ocr_res and ocr_res[0]:
            lines = [line[1][0] for line in ocr_res[0]]
            detected_text = clean_text("".join(lines))

        print(f"  Plate #{i}: {detected_text if detected_text else 'Unreadable'}")

        # 3. Draw Bounding Box & Label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        if detected_text:
            (tw, th), _ = cv2.getTextSize(detected_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(img, detected_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Save output visualization
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"result_{image_path.name}"
    cv2.imwrite(str(out_path), img)
    print(f"\nAnnotated image saved to: {out_path}")

if __name__ == "__main__":
    run_anpr(TEST_IMAGE)