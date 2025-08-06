import os
import cv2
import numpy as np
import random
from tqdm import tqdm

# Path to the base dataset folder
base_dir = "data"

def augment_image(img):
    # Flip horizontally
    if random.random() > 0.5:
        img = cv2.flip(img, 1)

    # Random rotation
    angle = random.uniform(-15, 15)
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Brightness adjustment
    brightness = random.uniform(0.8, 1.2)
    img = np.clip(img * brightness, 0, 255).astype(np.uint8)

    # Add Gaussian noise
    noise = np.random.normal(0, 10, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    return img

# Count images per emotion
emotion_counts = {}
for emotion in os.listdir(base_dir):
    emotion_path = os.path.join(base_dir, emotion)
    if os.path.isdir(emotion_path):
        emotion_counts[emotion] = len(os.listdir(emotion_path))

max_count = max(emotion_counts.values())
print("Class counts:", emotion_counts)
print("Balancing all to:", max_count)

# Augment each class
for emotion in os.listdir(base_dir):
    class_dir = os.path.join(base_dir, emotion)
    if not os.path.isdir(class_dir):
        continue

    images = os.listdir(class_dir)
    print(f"\nAugmenting '{emotion}': {len(images)} → {max_count}")

    while len(os.listdir(class_dir)) < max_count:
        img_name = random.choice(images)
        img_path = os.path.join(class_dir, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape != (48, 48):
            continue

        aug_img = augment_image(img)
        new_name = f"aug_{len(os.listdir(class_dir))}.png"
        cv2.imwrite(os.path.join(class_dir, new_name), aug_img)
