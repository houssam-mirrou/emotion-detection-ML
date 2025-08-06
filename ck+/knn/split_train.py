import os
import shutil
import random
from tqdm import tqdm

# Define paths
input_folder = "data/"  # The folder where images are currently stored
output_folder = "split_data/"  # Folder for train/test split

# Define split ratio
train_ratio = 0.8  # 80% training, 20% testing

# Create output directories
train_folder = os.path.join(output_folder, "train")
test_folder = os.path.join(output_folder, "test")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Iterate through emotion folders
for emotion in os.listdir(input_folder):
    emotion_path = os.path.join(input_folder, emotion)
    if not os.path.isdir(emotion_path):
        continue  # Skip non-folder files
    
    images = os.listdir(emotion_path)
    random.shuffle(images)  # Shuffle images for randomness

    # Split dataset
    train_count = int(len(images) * train_ratio)
    train_images = images[:train_count]
    test_images = images[train_count:]

    # Create subdirectories
    os.makedirs(os.path.join(train_folder, emotion), exist_ok=True)
    os.makedirs(os.path.join(test_folder, emotion), exist_ok=True)

    # Move images to train folder
    for img in tqdm(train_images, desc=f"Moving {emotion} images to train"):
        src = os.path.join(emotion_path, img)
        dest = os.path.join(train_folder, emotion, img)
        shutil.copy(src, dest)

    # Move images to test folder
    for img in tqdm(test_images, desc=f"Moving {emotion} images to test"):
        src = os.path.join(emotion_path, img)
        dest = os.path.join(test_folder, emotion, img)
        shutil.copy(src, dest)

print("Dataset successfully split into training and testing folders!")
