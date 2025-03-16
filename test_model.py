import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import datasets
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
model_path = "char_recognizer_model_ffnn.keras"
print(f"Loading model from {model_path}...")
model = load_model(model_path)

# Load MNIST test data
print("Loading MNIST test data...")
(_, _), (x_test, y_test) = datasets.mnist.load_data()

# Preprocess test data
x_test = x_test / 255.0  # Normalize to 0-1
x_test = x_test.reshape((-1, 28, 28))  # Keep shape as (28, 28) since model expects Flatten

# Evaluate model on test data
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Map labels to characters (0-9)
label_map = "0123456789"

# Function to display image and prediction
def display_prediction(image, true_label, pred_label):
    plt.imshow(image, cmap='gray')
    plt.title(f"True: {label_map[true_label]} | Pred: {label_map[pred_label]}")
    plt.axis('off')
    plt.show()

# Make predictions on random samples
num_samples = 5
indices = np.random.choice(len(x_test), num_samples, replace=False)

for idx in indices:
    img = x_test[idx]
    true_label = y_test[idx]
    pred = model.predict(np.expand_dims(img, axis=0))  # Model expects batch dimension
    pred_label = np.argmax(pred)

    display_prediction(img, true_label, pred_label)
