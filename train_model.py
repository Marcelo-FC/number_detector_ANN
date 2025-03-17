import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Desactiva optimizaciones OneDNN en TensorFlow para evitar problemas de compatibilidad.

from tensorflow.keras import datasets, layers, models, callbacks
import tensorflow as tf

# 📌 Habilitar crecimiento dinámico de memoria en GPU (evita errores por asignación de memoria fija)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Crecimiento dinámico de memoria habilitado para {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(e)

# 📥 Cargar el dataset MNIST (dígitos escritos a mano)
# MNIST contiene imágenes en escala de grises de 28x28 píxeles con números del 0 al 9.
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# 🔄 Normalizar los datos a valores entre 0 y 1 para mejorar la convergencia del entrenamiento.
x_train, x_test = x_train / 255.0, x_test / 255.0

# 🎯 Construcción del modelo FFNN (Feedforward Neural Network)
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Convierte la imagen 28x28 en un vector de 784 elementos.
    layers.Dense(128, activation="relu"),  # Capa oculta con 128 neuronas y activación ReLU (detecta patrones en la imagen).
    layers.Dense(64, activation="relu"),   # Segunda capa oculta con 64 neuronas para refinar la representación.
    layers.Dense(10, activation="softmax") # Capa de salida con 10 neuronas (una por cada dígito 0-9), activación softmax.
])

# 🛠️ Compilar el modelo
model.compile(
    optimizer="adam",  # Algoritmo de optimización basado en gradiente descendente adaptativo.
    loss="sparse_categorical_crossentropy",  # Función de pérdida para clasificación con etiquetas enteras (0-9).
    metrics=["accuracy"]  # Seguimiento de la precisión durante el entrenamiento.
)

# ⏳ Configuración de Early Stopping
# Detiene el entrenamiento si la pérdida en validación no mejora en 5 épocas consecutivas.
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',    # Observa la función de pérdida en validación.
    patience=5,            # Espera 5 épocas antes de detenerse si no hay mejoras.
    restore_best_weights=True  # Restaura los mejores pesos encontrados antes de la detención.
)

# 🚀 Entrenamiento del modelo usando GPU (si está disponible) con Early Stopping
with tf.device('/GPU:0'):
    model.fit(
        x_train, y_train,
        epochs=30,  # Entrena el modelo durante un máximo de 30 épocas.
        validation_split=0.2,  # Usa el 20% de los datos de entrenamiento para validación.
        callbacks=[early_stop]  # Usa el mecanismo de Early Stopping.
    )

# 📊 Evaluar el modelo con los datos de prueba (test)
loss, accuracy = model.evaluate(x_test, y_test)
print(f"📉 Pérdida: {loss:.4f}, 🎯 Precisión: {accuracy:.4f}")

# 💾 Guardar el modelo entrenado en formato Keras
model.save("char_recognizer_model_ffnn.keras")
print("✅ Modelo guardado como 'char_recognizer_model_ffnn.keras'")
