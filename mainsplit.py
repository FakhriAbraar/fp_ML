import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Muat dua model terpisah
age_model = load_model('age_model.h5', compile=False)
gender_model = load_model('gender_model.h5', compile=False)

# Buka kamera
cap = cv2.VideoCapture(0)

# Detektor wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi wajah
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (100, 100)) / 255.0
        face_input = np.expand_dims(face_resized, axis=0)

        # Prediksi gender dan usia
        gender_pred = gender_model.predict(face_input, verbose=0)
        age_pred = age_model.predict(face_input, verbose=0)

        pred_gender = np.argmax(gender_pred[0])
        gender_label = "Male" if pred_gender == 0 else "Female"
        age_label = int(age_pred[0][0])

        label = f"{gender_label}, Age: {age_label}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Real-Time Age & Gender (Dual Model)", frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()