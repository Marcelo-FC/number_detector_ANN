import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Desactiva optimizaciones OneDNN en TensorFlow Lite para evitar problemas de compatibilidad.

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
import matplotlib.pyplot as plt

# 📌 Ruta del modelo TFLite (versión optimizada en Float16)
model_path = "char_recognizer_model_ffnn_fp16.tflite"

# 📥 Verificar que el modelo existe antes de cargarlo
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Modelo TFLite no encontrado: {model_path}")

print(f"✅ Cargando modelo TFLite desde {model_path}...")
interpreter = tf.lite.Interpreter(model_path=model_path)  # Cargar modelo TFLite
interpreter.allocate_tensors()  # Asignar memoria para el modelo
print("✅ Modelo TFLite cargado y asignado correctamente.")

# 📊 Obtener detalles de entrada y salida del modelo
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 📥 Cargar el conjunto de datos de prueba MNIST
print("🔹 Cargando datos de prueba MNIST...")
(_, _), (x_test, y_test) = datasets.mnist.load_data()

# 🔄 Preprocesamiento de los datos de prueba
x_test = x_test / 255.0  # Normalización de píxeles a un rango entre 0 y 1
x_test = x_test.astype(np.float32)  # Asegurar que los datos sean float32 (requerido por el intérprete)
x_test = x_test.reshape((-1, 28, 28))  # Mantener forma (28, 28) ya que el modelo usa Flatten

# 📌 Diccionario para mapear etiquetas a caracteres del 0 al 9
label_map = "0123456789"

# 📌 Función para ejecutar inferencia en una imagen y obtener la predicción
def predict_tflite(image):
    """
    Ejecuta inferencia en una imagen utilizando el modelo TFLite.
    
    Parámetros:
    - image: Imagen preprocesada de tamaño (28,28).

    Retorna:
    - pred_label: Dígito predicho con mayor probabilidad.
    - output: Distribución de probabilidades para cada dígito (0-9).
    """
    # Agregar dimensión de lote y asegurar formato correcto para el intérprete
    image = np.expand_dims(image, axis=0).astype(np.float32)

    # Pasar la imagen al modelo
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()  # Ejecutar inferencia

    # Obtener resultados de la salida del modelo
    output = interpreter.get_tensor(output_details[0]['index'])

    # Determinar el dígito con la mayor probabilidad
    return np.argmax(output), output

# 📌 Función para mostrar la imagen junto con la predicción
def display_prediction(image, true_label, pred_label):
    """
    Muestra la imagen del dígito con su etiqueta real y su predicción.

    Parámetros:
    - image: Imagen en escala de grises del dígito.
    - true_label: Etiqueta real del dígito (valor correcto).
    - pred_label: Etiqueta predicha por el modelo.
    """
    plt.imshow(image, cmap='gray')  # Mostrar imagen en escala de grises
    plt.title(f"Verdadero: {label_map[true_label]} | Predicho: {label_map[pred_label]}")  # Etiqueta real y predicha
    plt.axis('off')  # Ocultar ejes
    plt.show()

# 📊 Evaluar la precisión del modelo TFLite en el conjunto de prueba
print("🔹 Evaluando el modelo TFLite en los datos de prueba...")
correct = 0  # Contador de predicciones correctas

for i in range(len(x_test)):  
    pred_label, _ = predict_tflite(x_test[i])  # Obtener predicción
    if pred_label == y_test[i]:  # Comparar con la etiqueta real
        correct += 1  

accuracy = correct / len(x_test)  # Calcular precisión
print(f"✅ Precisión del modelo TFLite en prueba: {accuracy:.4f}")

# 🔹 Realizar predicciones en muestras aleatorias del conjunto de prueba
print("🔹 Realizando predicciones en muestras aleatorias...")
num_samples = 5  # Número de muestras a evaluar visualmente
indices = np.random.choice(len(x_test), num_samples, replace=False)  # Seleccionar índices aleatorios

for idx in indices:
    img = x_test[idx]  # Obtener imagen de prueba
    true_label = y_test[idx]  # Obtener etiqueta real
    pred_label, _ = predict_tflite(img)  # Obtener predicción del modelo

    display_prediction(img, true_label, pred_label)  # Mostrar la predicción
