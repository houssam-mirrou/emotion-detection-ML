import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Parameter
image_size = (48, 48)
batch_size = 32
epochs = 100

# paths
train_dir = 'split_data/train'
test_dir = 'split_data/test'

# image generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)

# model construction
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1), padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6)

# training
start_time = time.time()
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=epochs,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
training_time = time.time() - start_time
print(f"\n[RESULT] Training Time : {training_time:.2f} seconds")

# model saving
model.save("model_file.h5")

# accuracy
loss, accuracy = model.evaluate(test_generator, verbose=0)
print(f"\n[RESULT] Accuracy on the testing set : {accuracy*100:.2f}%")

# inference time
start_time = time.time()
_ = model.predict(test_generator, verbose=0)
inference_time_per_sample = (time.time() - start_time) / len(test_generator.classes)
print(f"\n[RESULT] Inference Time per Image: : {inference_time_per_sample:.6f} seconds")

# memory usage
def get_size(path):
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

train_memory = get_size(train_dir) / (1024**2)  # en Mo
print(f"\n[RESULT] Memory usage on training set : {train_memory:.2f} Mo")

# prediction
y_pred = np.argmax(model.predict(test_generator, verbose=0), axis=1)
y_true = test_generator.classes

# prediction on training set (non-shufled)
train_generator_noshuffle = test_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)
y_train_pred = np.argmax(model.predict(train_generator_noshuffle, verbose=0), axis=1)
y_train_true = train_generator_noshuffle.classes

class_labels = list(train_generator.class_indices.keys())

print("\n[RESULT] Classification report (Train):")
print(classification_report(y_train_true, y_train_pred, target_names=class_labels))

print("\n[RESULT] Classification report (Test):")
print(classification_report(y_true, y_pred, target_names=class_labels))

# confusion matrix
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_train_true, y_train_pred), annot=True, fmt="d",
            xticklabels=class_labels, yticklabels=class_labels, cmap="Blues")
plt.title("Train Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Real")

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d",
            xticklabels=class_labels, yticklabels=class_labels, cmap="Oranges")
plt.title("Test Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Real")

plt.tight_layout()
plt.show()

# learning curve
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# noizy data
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
    target_size=image_size,
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)

noisy_loss, noisy_accuracy = model.evaluate(noisy_test_generator, verbose=0)
print(f"\n[RESULT] Robustness on Noisy Data: {noisy_accuracy*100:.2f}%")

# Summary of the model
model.summary()
total_params = model.count_params()
print(f"\n[RESULT] Total number of model parameters : {total_params:,}")
