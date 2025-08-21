import cv2
import face_recognition

# Load a known image and encode it
known_img = face_recognition.load_image_file("john.jpg")
known_encoding = face_recognition.face_encodings(known_img)[0]

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    rgb_frame = frame[:, :, ::-1]  # BGR → RGB

    # Find face locations and encodings in current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces([known_encoding], face_encoding)

        if True in matches:
            name = "John"
        else:
            name = "Unknown"

        # Draw box + label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Attendance Scanner", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
