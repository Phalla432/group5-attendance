import argparse
import os
import cv2
import pyttsx3
from ultralytics import YOLO

# --- Load reference images ---
def load_reference_images(folder="check_attendance"):
    references = {}
    orb = cv2.ORB_create()
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if not os.path.isfile(path):
            continue
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        kp, des = orb.detectAndCompute(img, None)
        references[file] = (kp, des)
    return references

# --- Match function ---
def is_match(des1, des2, threshold=20):
    if des1 is None or des2 is None:
        return False
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return len(matches) > threshold

# --- Text-to-speech function ---
def speak(message):
    engine = pyttsx3.init()
    engine.say(message)
    engine.runAndWait()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0")
    args = parser.parse_args()

    # Load YOLO model (detect people/faces depending on model)
    model = YOLO("yolov8n.pt")

    # Load reference pictures
    references = load_reference_images("check_attendance")
    orb = cv2.ORB_create()

    source = 0 if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    spoken = None  # Track last spoken message

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model.predict(frame, conf=0.5)
        for r in results:
            for box in r.boxes.xyxy:  # bounding boxes
                x1, y1, x2, y2 = map(int, box[:4])
                roi = frame[y1:y2, x1:x2]

                # Convert detected ROI to grayscale
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                kp, des = orb.detectAndCompute(gray, None)

                matched = False
                for name, (ref_kp, ref_des) in references.items():
                    if is_match(des, ref_des):
                        cv2.putText(frame, "You can join!", (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                        if spoken != "join":   # Prevent repeat
                            speak("You can join!")
                            spoken = "join"
                        matched = True
                        break

                if not matched:
                    cv2.putText(frame, "no no you can't join!", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                    if spoken != "nojoin":   # Prevent repeat
                        speak("no no you can't join!")
                        spoken = "nojoin"

                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)

        cv2.imshow("Attendance Check", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


