import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Desactiva optimizaciones de OneDNN para evitar problemas de compatibilidad.

import numpy as np
import tflite_runtime.interpreter as tflite  # ✅ Se usa tflite-runtime en lugar de TensorFlow completo para reducir el tamaño del contenedor.

# 📌 Ruta del modelo TFLite optimizado en Float16
model_path = "char_recognizer_model_ffnn_fp16.tflite"

# 📥 Verificar que el modelo existe antes de cargarlo
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Modelo TFLite no encontrado: {model_path}")

# 📥 Cargar el modelo TFLite
print(f"✅ Cargando modelo TFLite desde {model_path}...")
interpreter = tflite.Interpreter(model_path=model_path)  # Cargar el modelo usando TFLite Runtime
interpreter.allocate_tensors()  # Asignar memoria para la inferencia
print("✅ Modelo TFLite cargado y asignado correctamente.")

# 📊 Obtener detalles de entrada y salida del modelo
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 📌 Función para predecir un dígito a partir de una imagen preprocesada
def predict_digit(image_array):
    """
    Recibe una imagen en escala de grises con dimensiones (1, 28, 28) ya preprocesada.
    Devuelve el dígito predicho como una cadena de texto.
    
    Parámetros:
    - image_array: Imagen preprocesada en forma de un array numpy con valores normalizados.

    Retorna:
    - predicted_digit: Dígito predicho con mayor probabilidad en formato string.
    """

    # 📌 Asegurar que la imagen tiene el formato correcto para TFLite
    image_array = image_array.astype(np.float32)  # TFLite espera datos en float32, incluso en FP16

    # 📌 Asignar la imagen al tensor de entrada del modelo
    interpreter.set_tensor(input_details[0]['index'], image_array)

    # 📌 Ejecutar la inferencia con el modelo
    interpreter.invoke()

    # 📌 Obtener los resultados de salida (probabilidades de los 10 dígitos 0-9)
    output = interpreter.get_tensor(output_details[0]['index'])

    # 📌 Determinar el dígito con la mayor probabilidad
    predicted_digit = np.argmax(output)

    return str(predicted_digit)
