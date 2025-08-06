import os
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.feature import hog 
import matplotlib.pyplot as plt
import joblib
import time
import sys
import dlib
import seaborn as sns
from sklearn.model_selection import learning_curve

start_time = time.time()

# Initialisation du détecteur de visage DLIB
face_detector = dlib.get_frontal_face_detector()

# Configuration
IMG_SIZE = (48, 48)
train_folder = "split_data/train"
test_folder = "split_data/test"

emotion_labels = {
    "Anger": 0, "Contempt": 1, "Disgust": 2, "Fear": 3,
    "Happiness": 4, "Neutral": 5, "Sadness": 6, "Surprise": 7
}
label_to_emotion = {v: k for k, v in emotion_labels.items()}

# Fonction de détection et extraction HOG
def extract_face(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    dets = face_detector(img, 1)
    if len(dets) == 0:
        return None
    x, y, x2, y2 = dets[0].left(), dets[0].top(), dets[0].right(), dets[0].bottom()
    x, y = max(0, x), max(0, y)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    face = img[y:y2, x:x2]
    try:
        face = cv2.resize(face, IMG_SIZE)
    except:
        return None
    return hog(face, pixels_per_cell=(8,8), cells_per_block=(2,2), orientations=9)

# Chargement des images
def load_dataset(folder):
    X, y = [], []
    for emotion, label in emotion_labels.items():
        emotion_path = os.path.join(folder, emotion)
        if not os.path.isdir(emotion_path):
            continue
        for img_name in tqdm(os.listdir(emotion_path), desc=f"Processing {emotion}"):
            img_path = os.path.join(emotion_path, img_name)
            features = extract_face(img_path)
            if features is not None:
                X.append(features)
                y.append(label)
    return np.array(X), np.array(y)

# Chargement données
X_train, y_train = load_dataset(train_folder)
X_test, y_test = load_dataset(test_folder)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Entraînement du modèle k-NN
knn = KNeighborsClassifier(n_neighbors=31, weights="distance")
knn.fit(X_train, y_train)
training_time = time.time() - start_time
print(f"Training time: {training_time:.2f} s")

# Évaluation du modèle
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc * 100:.2f}%")

# Sauvegarde du modèle
joblib.dump(knn, 'knn_emotion_model.pkl')

# Temps d'inférence moyen
num_samples = min(925, len(X_test))
times = []
for i in range(num_samples):
    sample = X_test[i].reshape(1, -1)
    start = time.time()
    _ = knn.predict(sample)
    end = time.time()
    times.append(end - start)
print(f"Inference time per sample: {sum(times)/len(times):.6f} s")

# Taille mémoire estimée
print(f"Model size: {sys.getsizeof(knn)/1024:.2f} KB")
print(f"Training data memory: {(X_train.nbytes + y_train.nbytes)/1024/1024:.2f} MB")

# Robustesse au bruit
def add_noise(x, noise_level=0.1):
    return np.clip(x + np.random.normal(0, noise_level, x.shape), 0, 1)

X_test_noisy = np.array([add_noise(x) for x in X_test])
y_pred_noisy = knn.predict(X_test_noisy)
accuracy_noisy = accuracy_score(y_test, y_pred_noisy)
print(f"Accuracy on noisy data: {accuracy_noisy * 100:.2f}%")

print(f"[INFO] Model complexity at inference: O({len(X_train)} × {X_train.shape[1]})")


# ================================
# AJOUT COURBES + MATRICES + RAPPORTS
# ================================

emotion_names = list(emotion_labels.keys())

# Rapports de classification
print("\nClassification Report (Train):")
y_train_pred = knn.predict(X_train)
print(classification_report(y_train, y_train_pred, target_names=emotion_names))

print("\nClassification Report (Test):")
print(classification_report(y_test, y_pred, target_names=emotion_names))

# Matrices de confusion
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
train_cm = confusion_matrix(y_train, y_train_pred)
sns.heatmap(train_cm, annot=True, fmt="d", xticklabels=emotion_names, yticklabels=emotion_names, cmap="Blues")
plt.title("Train Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.subplot(1, 2, 2)
test_cm = confusion_matrix(y_test, y_pred)
sns.heatmap(test_cm, annot=True, fmt="d", xticklabels=emotion_names, yticklabels=emotion_names, cmap="Oranges")
plt.title("Test Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

train_sizes, train_scores, test_scores = learning_curve(
    knn, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=3)

train_scores_mean = train_scores.mean(axis=1)
test_scores_mean = test_scores.mean(axis=1)

plt.plot(train_sizes, train_scores_mean, label="Train")
plt.plot(train_sizes, test_scores_mean, label="Test")
plt.title("Learning Curve (k-NN)")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
