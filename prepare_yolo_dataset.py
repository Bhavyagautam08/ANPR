import csv
import hashlib
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
import cv2

ROOT = Path(r"D:\ANPR")
DATASET_1 = ROOT / "Dataset_1"
OUTPUT = ROOT / "yolo_combined_v1"

SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(SEED)
records = []

def get_dimensions(path):
    image = cv2.imread(str(path))
    if image is None:
        return None
    return image.shape[1], image.shape[0]

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

xml_files = sorted(DATASET_1.rglob("*.xml"))

for xml_path in xml_files:
    try:
        annotation = ET.parse(xml_path).getroot()
    except Exception:
        continue

    filename = annotation.findtext("filename", default="")
    image_path = xml_path.parent / Path(filename).name if filename else None
    
    if not image_path or not image_path.is_file():
        continue

    dims = get_dimensions(image_path)
    if not dims:
        continue

    width, height = dims
    boxes = []

    for obj in annotation.findall("object"):
        node = obj.find("bndbox")
        if node is None:
            continue

        try:
            xmin = float(node.findtext("xmin")) - 1
            ymin = float(node.findtext("ymin")) - 1
            xmax = float(node.findtext("xmax"))
            ymax = float(node.findtext("ymax"))
        except Exception:
            continue

        if 0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height:
            cx = (xmin + xmax) / (2 * width)
            cy = (ymin + ymax) / (2 * height)
            bw = (xmax - xmin) / width
            bh = (ymax - ymin) / height
            boxes.append((cx, cy, bw, bh))

    if boxes:
        records.append({
            "image": image_path,
            "boxes": boxes,
            "hash": sha256_file(image_path)
        })

# Split dataset (70% train, 15% val, 15% test)
random.shuffle(records)
n = len(records)
splits = {
    "train": records[:int(n * 0.7)],
    "val": records[int(n * 0.7):int(n * 0.85)],
    "test": records[int(n * 0.85):]
}

for split, items in splits.items():
    (OUTPUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT / "labels" / split).mkdir(parents=True, exist_ok=True)
    for rec in items:
        src = rec["image"]
        dest_img = OUTPUT / "images" / split / src.name
        dest_lbl = OUTPUT / "labels" / split / f"{src.stem}.txt"
        shutil.copy2(src, dest_img)
        with dest_lbl.open("w", encoding="utf-8") as f:
            for box in rec["boxes"]:
                f.write(f"0 {box[0]:.8f} {box[1]:.8f} {box[2]:.8f} {box[3]:.8f}\n")

(OUTPUT / "data.yaml").write_text(
    f"path: {OUTPUT.as_posix()}\n"
    "train: images/train\n"
    "val: images/val\n"
    "test: images/test\n\n"
    "names:\n"
    "  0: license_plate\n",
    encoding="utf-8"
)

print(f"Dataset ready at {OUTPUT}! Total images processed: {n}")