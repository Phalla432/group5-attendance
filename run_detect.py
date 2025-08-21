import argparse
import os
import cv2
from ultralytics import YOLO

def parse_args():
    ap = argparse.ArgumentParser(description="YOLOv8 detection (vehicles).")
    ap.add_argument("--source", type=str, default="0", help="Camera index or video path (use 0 for webcam)")
    ap.add_argument("--weights", type=str, default="", help="Path to weights. If empty, uses config or yolov8n.pt")
    ap.add_argument("--conf", type=float, default=None, help="Confidence threshold override")
    ap.add_argument("--iou", type=float, default=None, help="NMS IoU threshold override")
    ap.add_argument("--imgsz", type=int, default=None, help="Inference image size")
    ap.add_argument("--save", action="store_true", help="Save annotated video to outputs/")
    return ap.parse_args()

def load_config():
    import json, pathlib
    cfg_path = pathlib.Path(__file__).resolve().parents[1] / "config.json"
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            return json.load(f)
    return {}

def open_source(src_str):
    # Handle webcam index vs path
    if src_str.isdigit():
        return int(src_str)
    return src_str

def main():
    args = parse_args()
    cfg = load_config()

    model_path = args.weights or cfg.get("model", "")
    # Fallback to default small model if no custom weights present
    if not model_path or not os.path.isfile(model_path):
        model_path = "yolov8n.pt"  # downloaded automatically by Ultralytics

    conf = args.conf if args.conf is not None else cfg.get("conf", 0.25)
    iou = args.iou if args.iou is not None else cfg.get("iou", 0.45)
    imgsz = args.imgsz if args.imgsz is not None else cfg.get("imgsz", 640)

    model = YOLO(model_path)
    source = open_source(args.source)

    save_path = None
    writer = None
    if args.save:
        os.makedirs("outputs", exist_ok=True)
        # Prepare writer after we read first frame to know size
        cap = cv2.VideoCapture(0 if isinstance(source, int) else source)
        ok, frame = cap.read()
        if not ok:
            print("Could not read from source to initialize writer.")
            return
        h, w = frame.shape[:2]
        save_path = os.path.join("outputs", "detect_annotated.mp4")
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
        cap.release()

    for result in model.predict(source=source, conf=conf, iou=iou, imgsz=imgsz, stream=True):
        frame = result.plot()
        cv2.imshow("YOLOv8 Vehicle Detection", frame)
        if writer is not None:
            writer.write(frame)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    if writer is not None:
        writer.release()
        print(f"Saved: {save_path}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()