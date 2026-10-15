from pathlib import Path
from ultralytics import YOLO

ROOT = Path(r"D:\ANPR")
DATA_YAML = ROOT / "yolo_combined_v1" / "data.yaml"

if __name__ == "__main__":
    # Load pre-trained nano YOLO11 model
    model = YOLO("yolo11n.pt")

    # Train model on GPU
    model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,  # Uses your RTX 2050
        project=str(ROOT / "runs" / "plate_detection"),
        name="yolo11n_plate_run",
        exist_ok=True
    )