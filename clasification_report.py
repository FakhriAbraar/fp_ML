import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# === Load Gender Model ===
gender_model = load_model("gender_model.h5", compile=False)

# === Fungsi untuk Load Gambar dan Label Gender ===
def load_data(path, image_size=100, max_images=10000):
    images = []
    genders = []
    files = [f for f in os.listdir(path) if f.endswith(".jpg")]
    for f in files[:max_images]:
        try:
            age, gender = map(int, f.split("_")[:2])
            if gender not in [0, 1]:
                continue
            img_path = os.path.join(path, f)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (image_size, image_size))
            images.append(img)
            genders.append(gender)
        except:
            continue
    X = np.array(images) / 255.0
    y = to_categorical(np.array(genders), 2)
    return X, y

# === Load Dataset dan Split ===
X, y = load_data("dataset")  # Ganti jika path berbeda
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Prediksi ===
preds = gender_model.predict(X_test, verbose=0)
y_pred = np.argmax(preds, axis=1)
y_true = np.argmax(y_test, axis=1)

# === Classification Report ===
print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=["Female", "Male"]))

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred)
labels = ["Female", "Male"]

# Visualisasi Confusion Matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix - Gender Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
plt.savefig("confusion_matrix_gender.png")
