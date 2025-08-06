import os
import shutil
import random
from tqdm import tqdm

# Set random seed for reproducibility
random.seed(42)

# Define paths
input_folder = "data/"           # Folder with original images per emotion
output_folder = "split_data/"    # Output for train/test folders

# Define split ratio
train_ratio = 0.8

# Output subdirectories
train_folder = os.path.join(output_folder, "train")
test_folder = os.path.join(output_folder, "test")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Iterate through each emotion class folder
for emotion in os.listdir(input_folder):
    emotion_path = os.path.join(input_folder, emotion)
    if not os.path.isdir(emotion_path):
        continue  # Skip any non-folder files

    images = [img for img in os.listdir(emotion_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(images)

    if total_images < 2:
        print(f"Skipping '{emotion}' because it has fewer than 2 images.")
        continue

    # Shuffle to ensure randomness
    random.shuffle(images)

    # Calculate train/test split
    train_count = max(1, int(train_ratio * total_images))  # Ensure at least 1 image
    test_count = total_images - train_count

    # Fix edge case: if test_count becomes 0
    if test_count == 0:
        train_count -= 1
        test_count = 1

    train_images = images[:train_count]
    test_images = images[train_count:]

    # Create output subfolders
    os.makedirs(os.path.join(train_folder, emotion), exist_ok=True)
    os.makedirs(os.path.join(test_folder, emotion), exist_ok=True)

    # Copy training images
    for img in tqdm(train_images, desc=f"[{emotion}] -> train"):
        src = os.path.join(emotion_path, img)
        dst = os.path.join(train_folder, emotion, img)
        shutil.copy2(src, dst)

    # Copy testing images
    for img in tqdm(test_images, desc=f"[{emotion}] -> test"):
        src = os.path.join(emotion_path, img)
        dst = os.path.join(test_folder, emotion, img)
        shutil.copy2(src, dst)

print("\n✅ Dataset successfully split with class-aware (stratified-like) logic!")
