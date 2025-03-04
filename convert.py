# Run this locally:
from tensorflow.keras.models import load_model

model = load_model("char_recognizer_emnist_cnn.keras")
model.save("char_recognizer_emnist_cnn.h5")