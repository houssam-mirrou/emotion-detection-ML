# === 1. Imports and Configuration (if not already done earlier) ===
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# === 2. Constants ===
IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 120
model_path = "best_model.h5"
history_file = "training_history.pkl"

train_dir = 'Data/train/'
test_dir = 'Data/test/'
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# === 3. Data Generators ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)
val_data_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    color_mode='grayscale',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    classes=class_labels
)

validation_generator = val_data_gen.flow_from_directory(
    test_dir,
    color_mode='grayscale',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    classes=class_labels
)

# === 4. Class Weights ===
y_train = train_generator.classes
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(zip(np.unique(y_train), class_weights))

# === 5. Model Architecture ===
model = Sequential([
    Input(shape=(48, 48, 1)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.35),

    Conv2D(512, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),

    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(len(class_labels), activation='softmax')
])

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# === 6. Callbacks ===
checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1)
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-7, verbose=1)

# === 7. Training ===
print("Starting training...")
start_time = time.time()

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop, lr_reducer],
    class_weight=class_weights_dict,
    verbose=1
)

end_time = time.time()
print(f"Training Time: {end_time - start_time:.2f} seconds")

# === 8. Save History ===
with open(history_file, 'wb') as f:
    pickle.dump(history.history, f)

# === 9. Accuracy/Loss Curves ===
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy Curves')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss Curves')
plt.legend()
plt.tight_layout()
plt.show()

# === 10. Evaluation ===
validation_generator.reset()
y_pred = model.predict(validation_generator)
y_pred = np.argmax(y_pred, axis=1)
y_true = validation_generator.classes

print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=class_labels))

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.tight_layout()
plt.show()

# === 11. Robustness to Noise (optional) ===
def add_noise(img):
    noise_factor = 0.2
    noisy_img = img + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=img.shape)
    return np.clip(noisy_img, 0., 1.)

noisy_test_datagen = ImageDataGenerator(
    preprocessing_function=add_noise,
    rescale=1./255
)

noisy_test_generator = noisy_test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)

noisy_loss, noisy_accuracy = model.evaluate(noisy_test_generator, verbose=0)
print(f"\n[RESULT] Robustness on Noisy Data: {noisy_accuracy * 100:.2f}%")

# === 12. Summary ===
model.summary()
print(f"\n[RESULT] Total number of model parameters: {model.count_params():,}")
