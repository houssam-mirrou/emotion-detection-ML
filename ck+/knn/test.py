import cv2
import dlib
import numpy as np
import joblib
from skimage.feature import hog
import matplotlib.pyplot as plt

# --- Configuration ---
IMG_SIZE = (48, 48)
model_file = "knn_emotion_model.pkl"

label_to_emotion = {
    "Anger": 0, "Contempt": 1, "Disgust": 2, "Fear": 3,
    "Hapiness": 4, "Neutral": 5, "Sadness": 6, "Surprise": 7
}

# --- Chargement du modèle KNN ---
try:
    knn = joblib.load(model_file)
except Exception as e:
    print(f"[ERROR] Could not load model: {e}")
    exit()

# --- Détecteur de visage et landmarks ---
face_detector = dlib.get_frontal_face_detector()



# --- Fonction de détection du visage + extraction HOG ---
def extract_face(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("[ERROR] Failed to load image.")
        return None

    dets = face_detector(img, 1)
    if len(dets) == 0:
        print("[WARNING] No face detected.")
        return None

    face_rect = dets[0]
    x, y, x2, y2 = face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()
    face = img[y:y2, x:x2]

    try:
        face = cv2.resize(face, IMG_SIZE)
    except:
        print("[ERROR] Cropped face is invalid.")
        return None

    # Extraction HOG
    hog_features = hog(face, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9)
    return hog_features


# --- Classification ---
def classify_image(image_path, model):
    hog_vector = extract_face(image_path)
    if hog_vector is None:
        return None

    hog_vector = np.array(hog_vector).reshape(1, -1)
    predicted_label = model.predict(hog_vector)
    return label_to_emotion.get(predicted_label[0], "Unknown")


# --- Visualisation du visage détecté (dlib) ---
def visualize_faces(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("[ERROR] Cannot load image for visualization.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)

    for face in faces:
        x, y, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)

    if len(faces) == 0:
        print("[INFO] No face found to visualize.")
    else:
        cv2.imshow("Detected Faces", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# --- Visualisation des caractéristiques HOG (avec meilleure qualité) ---
def hog_presentation(img_url):
    image = cv2.imread(img_url, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("[ERROR] Cannot load image for HOG.")
        return

    resized_img = cv2.resize(image, (200, 200))  # Pour visualisation seulement
    hog_features, hog_image = hog(resized_img,
                                  pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2),
                                  orientations=9,
                                  visualize=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(resized_img, cmap="gray")
    ax1.set_title("Original image")

    ax2.imshow(hog_image, cmap="gray", interpolation='bicubic')
    ax2.set_title("HOG Features")
    plt.tight_layout()
    plt.show()


# --- Affichage de l'émotion prédite sur l'image ---
def display_prediction_on_image(image_path, emotion):
    img = cv2.imread(image_path)
    if img is None:
        print("[ERROR] Could not load image for display.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)

    if len(faces) == 0:
        print("[INFO] No face found to display prediction.")
        return

    d = faces[0]
    x, y = d.left(), d.top()
    text_position = (x, y + 200)

    # Afficher l'émotion prédite sur l'image originale
    cv2.putText(img, f"{emotion}", text_position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)

    cv2.imshow("Prediction Display", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --- Détection des landmarks faciaux ---
def detect_landmarks(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("[ERROR] Could not load image.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)

    if len(faces) == 0:
        print("[INFO] No face detected for landmarks.")
        return

    for face in faces:
        shape = predictor(gray, face)
        for i in range(shape.num_parts):
            part = shape.part(i)
            cv2.circle(img, (part.x, part.y), 2, (0, 255, 0), -1)
            cv2.putText(img, str(i), (part.x + 4, part.y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    cv2.imshow("Facial Landmarks", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --- Extraction des caractéristiques landmarks ---
def extract_landmark_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("[ERROR] Could not load image.")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)

    if len(faces) == 0:
        print("[INFO] No face detected for landmark features.")
        return None

    shape = predictor(gray, faces[0])

    left_eyebrow = (shape.part(19).x, shape.part(19).y)
    right_eyebrow = (shape.part(24).x, shape.part(24).y)
    left_eye = (shape.part(36).x, shape.part(36).y)
    right_eye = (shape.part(45).x, shape.part(45).y)
    mouth_left = (shape.part(48).x, shape.part(48).y)
    mouth_right = (shape.part(54).x, shape.part(54).y)

    def distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    eyebrow_distance = distance(left_eyebrow, right_eyebrow)
    eye_distance = distance(left_eye, right_eye)
    mouth_width = distance(mouth_left, mouth_right)

    features = np.array([eyebrow_distance, eye_distance, mouth_width])
    return features


# --- Test ---
if __name__ == "__main__":
    test_image = "surprise.png"  # Remplace par ton image réelle

    # Étape 1 : Visualiser les visages détectés
    visualize_faces(test_image)

    # Étape 2 : Classer l'image
    emotion = classify_image(test_image, knn)

    # Étape 3 : Visualisation des HOG Features
    hog_presentation(test_image)


    if emotion:
        print(f"[RESULT] Predicted Emotion: {emotion}")

        # Étape 4 : Afficher l'émotion sur l'image
        display_prediction_on_image(test_image, emotion)
    else:
        print("[RESULT] Could not detect emotion.")

