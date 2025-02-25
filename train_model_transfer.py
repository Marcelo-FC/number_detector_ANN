import os
import numpy as np
from scipy.io import loadmat
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping

# Enable dynamic memory growth for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Enabled memory growth for {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(e)

# Load EMNIST ByClass from the extracted .mat file
emnist_path = os.path.expanduser("~/.emnist/matlab/emnist-byclass.mat")
print(f"Loading EMNIST data from {emnist_path}...")
data = loadmat(emnist_path)

# Extract train/test images and labels
x_train = data['dataset']['train'][0, 0]['images'][0, 0]
y_train = data['dataset']['train'][0, 0]['labels'][0, 0].flatten()
x_test = data['dataset']['test'][0, 0]['images'][0, 0]
y_test = data['dataset']['test'][0, 0]['labels'][0, 0].flatten()

# Preprocess: Normalize and reshape for batching
x_train = x_train.reshape((-1, 28, 28), order='F') / 255.0
x_test = x_test.reshape((-1, 28, 28), order='F') / 255.0

# Function to resize images in batches
def preprocess_image(image, label):
    image = tf.expand_dims(image, -1)  # Add channel dimension
    image = tf.image.resize(image, (96, 96))  # Resize to 96x96
    image = tf.image.grayscale_to_rgb(image)  # Convert to RGB for MobileNetV2
    return image, label

# Create TensorFlow datasets for efficient processing
batch_size = 128
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = test_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Build Transfer Learning Model using MobileNetV2
base_model = MobileNetV2(input_shape=(96, 96, 3),
                         include_top=False,
                         weights='imagenet')
base_model.trainable = False  # Freeze base model

# Add custom layers
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(62, activation='softmax')  # 62 classes for EMNIST
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# EarlyStopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Train model
print("Starting Transfer Learning Training on GPU...")
history = model.fit(
    train_ds,
    epochs=20,
    validation_data=test_ds,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate model
loss, accuracy = model.evaluate(test_ds)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Save trained model
model.save("char_recognizer_emnist_transfer.keras")
print("Transfer Learning Model saved as 'char_recognizer_emnist_transfer.keras'.")
