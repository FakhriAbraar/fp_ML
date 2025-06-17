# === Bagian Import Tetap Sama ===
import os
import cv2
import numpy as np
import albumentations as A
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence

# === Fungsi Resize dan Load Data Tetap Sama ===
def resize_with_padding(img, target_size):
    old_size = img.shape[:2]
    ratio = float(target_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    resized_img = cv2.resize(img, (new_size[1], new_size[0]))
    delta_w = target_size - new_size[1]
    delta_h = target_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_img

def load_utkface_data(dataset_path, image_size=100, max_images=10000):
    images, ages, genders = [], [], []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    image_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
    for filename in image_files[:max_images]:
        try:
            age, gender = map(int, filename.split("_")[:2])
            if gender not in [0, 1]: continue
            img_path = os.path.join(dataset_path, filename)
            img = cv2.imread(img_path)
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) == 0: continue
            x, y, w, h = faces[0]
            face_img = img[y:y+h, x:x+w]
            face_img = resize_with_padding(face_img, image_size)
            images.append(face_img)
            ages.append(age)
            genders.append(gender)
        except: continue
    return np.array(images), np.array(ages), np.array(genders)

# === Augmentasi ===
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.4),
    A.MotionBlur(p=0.2),
    A.RandomGamma(p=0.3),
    A.ToGray(p=0.2)
])

# === Generator Data ===
class AugmentedDataGenerator(Sequence):
    def __init__(self, X, y, batch_size=32, augment=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.indices = np.arange(len(X))

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_X = self.X[batch_idx]
        batch_y = self.y[batch_idx]

        if self.augment:
            augmented_X = []
            for img in batch_X:
                aug = augmentation(image=img)
                augmented_X.append(aug['image'])
            batch_X = np.array(augmented_X)

        batch_X = batch_X / 255.0
        return batch_X, batch_y

# === Load Data ===
dataset_path = "dataset"
image_size = 100
X, age_labels, gender_labels = load_utkface_data(dataset_path, image_size=image_size)

# Format label
age_labels = age_labels.astype('float32')
gender_labels = to_categorical(gender_labels, 2)

# Split
X_train, X_test, age_train, age_test, gender_train, gender_test = train_test_split(
    X, age_labels, gender_labels, test_size=0.2, random_state=42
)

# Generator untuk masing-masing model
train_age_gen = AugmentedDataGenerator(X_train, age_train, augment=True)
val_age_gen = AugmentedDataGenerator(X_test, age_test, augment=False)
train_gender_gen = AugmentedDataGenerator(X_train, gender_train, augment=True)
val_gender_gen = AugmentedDataGenerator(X_test, gender_test, augment=False)

# === Model Age ===
input_age = Input(shape=(image_size, image_size, 3))
x_age = Conv2D(32, (3,3), activation='relu')(input_age)
x_age = MaxPooling2D()(x_age)
x_age = Conv2D(64, (3,3), activation='relu')(x_age)
x_age = MaxPooling2D()(x_age)
x_age = Flatten()(x_age)
age_output = Dense(1)(x_age)
model_age = Model(input_age, age_output)
model_age.compile(optimizer=Adam(1e-4), loss='mse', metrics=['mae'])
model_age.fit(train_age_gen, validation_data=val_age_gen, epochs=10)
model_age.save("age_model.h5")

# === Model Gender ===
input_gender = Input(shape=(image_size, image_size, 3))
x_gen = Conv2D(32, (3,3), activation='relu')(input_gender)
x_gen = MaxPooling2D()(x_gen)
x_gen = Conv2D(64, (3,3), activation='relu')(x_gen)
x_gen = MaxPooling2D()(x_gen)
x_gen = Flatten()(x_gen)
gender_output = Dense(2, activation='softmax')(x_gen)
model_gender = Model(input_gender, gender_output)
model_gender.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model_gender.fit(train_gender_gen, validation_data=val_gender_gen, epochs=10)
model_gender.save("gender_model.h5")