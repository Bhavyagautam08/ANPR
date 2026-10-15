import cv2
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from ultralytics import YOLO

# Suppress debug logs
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("paddle").setLevel(logging.ERROR)

# -------------------- CONFIGURATION --------------------
WEIGHTS_PATH = Path(r"D:\ANPR\runs\plate_detection\yolo11n_plate_run\weights\best.pt")
TEST_IMAGES_DIR = Path(r"D:\ANPR\Dataset_1\State-wise_OLX") 

print("Loading YOLO11 (GPU) and PaddleOCR (CPU)...")
yolo_model = YOLO(str(WEIGHTS_PATH))

from paddleocr import PaddleOCR
ocr_model = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)

def post_process_text(text):
    """
    Cleans OCR output and enforces standard Indian license plate formatting
    to automatically correct common B/8, 0/O, and 1/I confusions.
    """
    if not text:
        return ""
        
    # Remove all non-alphanumeric noise and strip stray "IND" watermarks
    text = "".join(ch for ch in text.upper() if ch.isalnum())
    text = text.replace("IND", "").replace("INDIA", "")
    
    if len(text) < 6:
        return text
        
    char_to_letter = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '8': 'B'}
    char_to_number = {'O': '0', 'I': '1', 'Z': '2', 'A': '4', 'S': '5', 'B': '8', 'G': '6'}
    
    fixed = list(text)
    
    # 1. State Code (First 2 chars MUST be letters)
    for i in range(min(2, len(fixed))):
        if fixed[i] in char_to_letter:
            fixed[i] = char_to_letter[fixed[i]]
            
    # 2. RTO Code (Next 2 chars MUST be numbers)
    for i in range(2, min(4, len(fixed))):
        if fixed[i] in char_to_number:
            fixed[i] = char_to_number[fixed[i]]
            
    # 3. Serial Number (Last 4 chars MUST be numbers)
    for i in range(max(4, len(fixed)-4), len(fixed)):
        if fixed[i] in char_to_number:
            fixed[i] = char_to_number[fixed[i]]
            
    return "".join(fixed)

def levenshtein_distance(s1, s2):
    """Calculates edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def get_true_text(image_path):
    """Extracts ground truth license plate text from XML."""
    xml_path = image_path.with_suffix('.xml')
    if not xml_path.exists():
        return ""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        obj = root.find('object')
        if obj is not None:
            name_tag = obj.find('name')
            if name_tag is not None and name_tag.text:
                return post_process_text(name_tag.text)
    except Exception as e:
        print(f"Error reading {xml_path}: {e}")
    return ""

def evaluate():
    image_paths = list(TEST_IMAGES_DIR.rglob("*.jpg"))
    if not image_paths:
        print(f"No images found in {TEST_IMAGES_DIR}")
        return

    total_images = 0
    exact_matches = 0
    total_edits = 0
    total_chars = 0

    print(f"\nEvaluating {len(image_paths)} images...\n")

    for img_path in image_paths:
        true_text = get_true_text(img_path)
        if not true_text:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # 1. Detect Plate Bounding Box (Restored to 0.5 confidence)
        results = yolo_model(img, conf=0.5)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        
        predicted_text = ""
        if len(boxes) > 0:
            x1, y1, x2, y2 = map(int, boxes[0][:4])
            
            # 10% margin padding around box
            mx, my = int((x2 - x1) * 0.10), int((y2 - y1) * 0.10)
            cx1, cy1 = max(0, x1 - mx), max(0, y1 - my)
            cx2, cy2 = min(w, x2 + mx), min(h, y2 + my)
            
            plate_crop = img[cy1:cy2, cx1:cx2]
            
            # 2. Run OCR directly on raw crop (NO CLAHE)
            ocr_res = ocr_model.ocr(plate_crop, cls=True)
            
            if ocr_res and ocr_res[0]:
                lines_text = [line[1][0] for line in ocr_res[0]]
                predicted_text = post_process_text("".join(lines_text))

        # 3. Metrics Calculation
        distance = levenshtein_distance(predicted_text, true_text)
        total_edits += distance
        total_chars += len(true_text)
        
        if predicted_text == true_text:
            exact_matches += 1
            match_status = "✓"
        else:
            match_status = "✗"
            
        total_images += 1
        print(f"[{match_status}] Target: {true_text:15} | Pred: {predicted_text:15} | Edits: {distance}")

    # Results Output
    accuracy = (exact_matches / total_images) * 100 if total_images > 0 else 0
    cer = (total_edits / total_chars) * 100 if total_chars > 0 else 0

    print("\n" + "="*40)
    print("ANPR PIPELINE EVALUATION RESULTS")
    print("="*40)
    print(f"Total Images Evaluated : {total_images}")
    print(f"Exact Match Accuracy   : {accuracy:.2f}%")
    print(f"Character Error Rate   : {cer:.2f}%")
    print("="*40)

if __name__ == "__main__":
    evaluate()