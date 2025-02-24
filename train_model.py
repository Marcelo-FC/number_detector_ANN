import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from tensorflow.keras import datasets, layers, models

# 1. Cargar el conjunto de datos MNIST
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# 2. Normalizar los valores de píxeles entre 0 y 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3. Redimensionar para añadir el canal (1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# 4. Definir el modelo CNN
model = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),  # 10 clases para los dígitos 0-9
    ]
)

# Compilar el modelo
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Entrenar el modelo
model.fit(x_train, y_train, epochs=5)

# Evaluar el modelo
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Guardar el modelo
model.save("char_recognizer_model.keras")
