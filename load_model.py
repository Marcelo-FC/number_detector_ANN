import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model = load_model("char_recognizer_emnist_cnn.keras")

# Mapping for EMNIST ByClass (0–9, A–Z, a–z)
emnist_classes = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 
    'u', 'v', 'w', 'x', 'y', 'z'
]

def predict_digit(image_array):
    # Predict class index
    predictions = model.predict(image_array)
    predicted_index = np.argmax(predictions)

    # Map index to character
    predicted_char = emnist_classes[predicted_index]
    return predicted_char
