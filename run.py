import os
import face_recognition
import cv2
from fastapi import FastAPI
from datetime import datetime

app = FastAPI()

# Load known faces
known_face_encodings = []
known_face_names = []

for file in os.listdir("known_faces"):
    image = face_recognition.load_image_file(f"known_faces/{file}")
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append(os.path.splitext(file)[0])

@app.get("/scan")
def scan_attendance():
    cap = cv2.VideoCapture(0)  # Open camera
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return {"error": "Camera not available"}

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    detected_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        if True in matches:
            name = known_face_names[matches.index(True)]
            detected_names.append(name)

    if detected_names:
        return {
            "status": "success",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "present": detected_names
        }
    else:
        return {"status": "no match"}