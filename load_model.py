import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from tensorflow.keras.models import load_model
import numpy as np

# Cargar el modelo desde el archivo
model = load_model("char_recognizer_model.keras")


def predict_digit(image_array):
    # Predecir la clase (dígito) de la imagen
    predictions = model.predict(image_array)

    # Obtener el dígito con la mayor probabilidad
    predicted_digit = np.argmax(predictions)

    return str(predicted_digit)
