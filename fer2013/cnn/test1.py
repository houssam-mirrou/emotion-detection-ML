import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('model_file.h5')

# Load the Haar cascade for face detection
cascade_path = 'haarcascade_frontalface_default.xml'  # Update this path if necessary
facedetect = cv2.CascadeClassifier(cascade_path)

# Define emotion labels
labels_dict = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise"
}

# Read the image
image_path = "1.jpeg"  # Update this path if necessary
frame = cv2.imread(image_path)

# Check if the image was loaded successfully
if frame is None:
    print(f"Error: Unable to load image at {image_path}")
    exit()

# Convert the image to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)

# Process each detected face
for x, y, w, h in faces:
    sub_face_img = gray[y:y+h, x:x+w]  # Get the region of interest
    resized = cv2.resize(sub_face_img, (48, 48))  # Resize the image
    normalize = resized / 255.0  # Normalize the pixel values
    reshaped = np.reshape(normalize, (1, 48, 48, 1))  # Reshape for the model input
    
    # Make prediction
    result = model.predict(reshaped)
    label = np.argmax(result, axis=1)[0]  # Get the predicted label
    print(f"Predicted label: {label} - Emotion: {labels_dict[label]}")  # Print the predicted label

    # Draw rectangles and label on the image
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)  # Draw outer rectangle in red
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)  # Draw thicker outer rectangle in orange
    cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)  # Draw filled rectangle for label background
    cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # Add label text

# Display the resulting frame
cv2.imshow("frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
