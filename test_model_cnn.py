import os
import numpy as np
from scipy.io import loadmat
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Enable dynamic memory growth for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Enabled memory growth for {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(e)

# Load the trained model
model_path = "char_recognizer_emnist_cnn.keras"
print(f"Loading model from {model_path}...")
model = load_model(model_path)

# Load EMNIST ByClass test data
emnist_path = os.path.expanduser("~/.emnist/matlab/emnist-byclass.mat")
print(f"Loading EMNIST data from {emnist_path}...")
data = loadmat(emnist_path)

# Extract test images and labels
x_test = data['dataset']['test'][0, 0]['images'][0, 0]
y_test = data['dataset']['test'][0, 0]['labels'][0, 0].flatten()

# Preprocess test data
x_test = x_test.reshape((-1, 28, 28, 1), order='F') / 255.0  # 'F' for MATLAB format

# Evaluate model on test data
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Map labels to characters (0-9, A-Z, a-z)
label_map = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Function to display image and prediction
def display_prediction(image, true_label, pred_label):
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"True: {label_map[true_label]} | Pred: {label_map[pred_label]}")
    plt.axis('off')
    plt.show()

# Make predictions on random samples
num_samples = 5
indices = np.random.choice(len(x_test), num_samples, replace=False)

for idx in indices:
    img = x_test[idx]
    true_label = y_test[idx]
    pred = model.predict(np.expand_dims(img, axis=0))
    pred_label = np.argmax(pred)

    display_prediction(img, true_label, pred_label)
