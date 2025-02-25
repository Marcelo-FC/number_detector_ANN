import os
import numpy as np
from scipy.io import loadmat
import tensorflow as tf
from tensorflow.keras import layers, models
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

# Preprocess data: Normalize and reshape
x_train = x_train.reshape((-1, 28, 28), order='F') / 255.0  # 'F' due to MATLAB format
x_test = x_test.reshape((-1, 28, 28), order='F') / 255.0

# Build Enhanced FFNN model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(512, activation='relu'),   # Increased neurons
    layers.Dropout(0.3),                    # Added Dropout for regularization
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dense(62, activation='softmax')  # 62 classes for EMNIST
])


# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model explicitly on GPU if available
print("Starting training on GPU...")



# Define EarlyStopping callback
early_stop = EarlyStopping(
    monitor='val_accuracy',  # Monitor validation accuracy
    patience=5,              # Stop after 5 epochs without improvement
    restore_best_weights=True # Rollback to best weights after stopping
)

# Train model with EarlyStopping on GPU
with tf.device('/GPU:0'):
    history = model.fit(
        x_train, y_train,
        epochs=50,               # Allow more epochs for better learning
        batch_size=128,
        validation_split=0.1,    # Use 10% of data for validation
        callbacks=[early_stop],  # Include EarlyStopping callback
        verbose=1
    )


# Evaluate model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Save trained model
model.save("char_recognizer_emnist_ffnn.keras")
print("Model saved as 'char_recognizer_emnist_ffnn.keras'.")
