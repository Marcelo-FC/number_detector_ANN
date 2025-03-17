import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Desactiva optimizaciones OneDNN en TensorFlow para evitar problemas de compatibilidad.

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import datasets
import matplotlib.pyplot as plt

# 📌 Habilitar crecimiento dinámico de memoria en GPU (evita problemas de asignación fija).
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Crecimiento dinámico de memoria habilitado para {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(e)

# 📥 Cargar el modelo previamente entrenado
model_path = "char_recognizer_model_ffnn.keras"
print(f"🔹 Cargando modelo desde {model_path}...")
model = load_model(model_path)  # Carga el modelo de Keras

# 📥 Cargar el conjunto de datos de prueba MNIST (dígitos escritos a mano)
print("📥 Cargando datos de prueba MNIST...")
(_, _), (x_test, y_test) = datasets.mnist.load_data()

# 🔄 Preprocesamiento de datos de prueba
x_test = x_test / 255.0  # Normalización de los valores de píxeles a un rango entre 0 y 1
x_test = x_test.reshape((-1, 28, 28))  # Mantener la forma (28, 28) ya que la primera capa es Flatten

# 📊 Evaluar el modelo en los datos de prueba
loss, accuracy = model.evaluate(x_test, y_test)
print(f"📉 Pérdida en prueba: {loss:.4f}, 🎯 Precisión en prueba: {accuracy:.4f}")

# 📌 Diccionario para mapear etiquetas a caracteres del 0 al 9
label_map = "0123456789"

# 📌 Función para mostrar una imagen junto con su predicción
def display_prediction(image, true_label, pred_label):
    """
    Muestra la imagen de un dígito con su etiqueta real y su predicción.
    
    Parámetros:
    - image: Imagen en escala de grises del dígito.
    - true_label: Etiqueta real del dígito (valor correcto).
    - pred_label: Etiqueta predicha por el modelo.
    """
    plt.imshow(image, cmap='gray')  # Mostrar la imagen en escala de grises
    plt.title(f"Verdadero: {label_map[true_label]} | Predicho: {label_map[pred_label]}")  # Etiqueta real y predicha
    plt.axis('off')  # Ocultar ejes
    plt.show()

# 🔹 Realizar predicciones en muestras aleatorias del conjunto de prueba
num_samples = 5  # Número de muestras a evaluar visualmente
indices = np.random.choice(len(x_test), num_samples, replace=False)  # Seleccionar índices aleatorios

for idx in indices:
    img = x_test[idx]  # Obtener imagen de prueba
    true_label = y_test[idx]  # Obtener la etiqueta real
    pred = model.predict(np.expand_dims(img, axis=0))  # Agregar dimensión batch para la predicción
    pred_label = np.argmax(pred)  # Obtener la clase con mayor probabilidad

    display_prediction(img, true_label, pred_label)  # Mostrar la predicción
