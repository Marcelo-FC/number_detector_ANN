import os
import tensorflow as tf

# 📌 Ruta del modelo Keras previamente entrenado
model_path = "char_recognizer_model_ffnn.keras"

# Verifica si el modelo existe antes de cargarlo
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ No se encontró el archivo del modelo: {model_path}")

# 🔹 Cargar el modelo Keras entrenado
model = tf.keras.models.load_model(model_path)
print("✅ Modelo Keras cargado correctamente.")

# 📌 Función para guardar modelos TFLite de manera segura
def save_tflite_model(tflite_model, filename):
    """
    Guarda el modelo en formato TFLite solo si no existe previamente.

    Parámetros:
    - tflite_model: Modelo convertido a formato TensorFlow Lite.
    - filename: Nombre del archivo donde se guardará el modelo.
    """
    if os.path.exists(filename):
        print(f"⚠️ Omitiendo {filename}, ya existe un archivo con ese nombre.")
    else:
        with open(filename, "wb") as f:
            f.write(tflite_model)
        print(f"✅ Modelo guardado: {filename}")

# 🔹 Conversión estándar a TensorFlow Lite (precisión Float32)
print("🔹 Convirtiendo a TFLite (Float32)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)  # Crear conversor
tflite_model = converter.convert()  # Convertir modelo
save_tflite_model(tflite_model, "char_recognizer_model_ffnn.tflite")  # Guardar

# 🔹 Conversión a TensorFlow Lite con cuantización en Float16
# Beneficios: Reduce el tamaño del modelo, mantiene buena precisión.
print("🔹 Convirtiendo a TFLite (cuantización Float16)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)  # Nuevo conversor
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Activar optimizaciones
converter.target_spec.supported_types = [tf.float16]  # Cuantización en 16 bits
tflite_fp16_model = converter.convert()  # Convertir modelo
save_tflite_model(tflite_fp16_model, "char_recognizer_model_ffnn_fp16.tflite")  # Guardar

# 📌 Mensaje final con los modelos generados
print("🎯 ¡Conversión completada!")
print("✅ Modelos generados:")
print("- char_recognizer_model_ffnn.tflite (Float32)")
print("- char_recognizer_model_ffnn_fp16.tflite (Cuantizado en Float16)")
