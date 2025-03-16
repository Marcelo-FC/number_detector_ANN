import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
import matplotlib.pyplot as plt

# Load the TFLite model
model_path = "char_recognizer_model_ffnn_fp16.tflite"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ TFLite model not found: {model_path}")

print(f"✅ Loading TFLite model from {model_path}...")
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
print("✅ TFLite model loaded and allocated successfully!")

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load MNIST test data
print("🔹 Loading MNIST test data...")
(_, _), (x_test, y_test) = datasets.mnist.load_data()

# Preprocess test data
x_test = x_test / 255.0  # Normalize to 0-1
x_test = x_test.astype(np.float32)  # Ensure float32 input type for interpreter
x_test = x_test.reshape((-1, 28, 28))  # Keep shape as (28, 28) since Flatten was used

# Map labels to characters (0-9)
label_map = "0123456789"

# Function to run inference on one image and get prediction
def predict_tflite(image):
    # Add batch dimension and adjust shape to interpreter input shape
    image = np.expand_dims(image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output), output

# Function to display image and prediction
def display_prediction(image, true_label, pred_label):
    plt.imshow(image, cmap='gray')
    plt.title(f"True: {label_map[true_label]} | Pred: {label_map[pred_label]}")
    plt.axis('off')
    plt.show()

# Evaluate model accuracy on test data
print("🔹 Evaluating TFLite model on test data...")
correct = 0
for i in range(len(x_test)):
    pred_label, _ = predict_tflite(x_test[i])
    if pred_label == y_test[i]:
        correct += 1

accuracy = correct / len(x_test)
print(f"✅ TFLite Test Accuracy: {accuracy:.4f}")

# Make predictions on random samples
print("🔹 Making predictions on random samples...")
num_samples = 5
indices = np.random.choice(len(x_test), num_samples, replace=False)

for idx in indices:
    img = x_test[idx]
    true_label = y_test[idx]
    pred_label, _ = predict_tflite(img)

    display_prediction(img, true_label, pred_label)
