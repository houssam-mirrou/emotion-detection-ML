import cv2
import os
import random

def augment_image(image):
    """Augment the image by flipping it."""
    flipped = cv2.flip(image, 1)  # Flip horizontally
    return flipped

def augment_class_images(class_folder, target_count):
    """Augment images in the specified folder until the target count is reached."""
    images = os.listdir(class_folder)
    
    # Count the number of current images
    current_count = len(images)
    print(f"Current count in {class_folder}: {current_count}")

    while current_count < target_count:
        img_name = random.choice(images)  # Randomly select an image
        img_path = os.path.join(class_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply augmentation
        augmented_img = augment_image(img)
        
        # Save augmented image with a new name
        new_name = f"aug_{random.randint(10000, 99999)}_{img_name}"
        cv2.imwrite(os.path.join(class_folder, new_name), augmented_img)
        
        # Increment the count after saving the augmented image
        current_count += 1

    print(f"Augmentation complete for {class_folder}. Total images: {current_count}")

# Define your class folder paths and target count
class_folders = [
    "data/Anger",      # Example for Anger class
    "data/Disgust",    # Example for Disgust class
    "data/Fear",       # Example for Fear class
    "data/Happiness",   # Example for Happiness class
    "data/Sadness",    # Example for Sadness class
    "data/Surprise",   # Example for Surprise class
    "data/Contempt"    # Example for Contempt class
]

# Target count (same as Neutral class count)
target_count = 593  

# Call the augmentation function for each class
for class_folder in class_folders:
    augment_class_images(class_folder, target_count)
