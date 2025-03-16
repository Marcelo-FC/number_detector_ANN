import os
import tensorflow as tf

# ✅ Load the trained MNIST model
model_path = "char_recognizer_model_ffnn.keras"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found: {model_path}")

model = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully!")

# Function to save TFLite model safely
def save_tflite_model(tflite_model, filename):
    """Save TFLite model only if it does not already exist."""
    if os.path.exists(filename):
        print(f"⚠️ Skipping {filename}, file already exists.")
    else:
        with open(filename, "wb") as f:
            f.write(tflite_model)
        print(f"✅ Saved: {filename}")

# 🔹 Standard TFLite Conversion (Float32)
print("🔹 Converting to TFLite (Float32)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
save_tflite_model(tflite_model, "char_recognizer_model_ffnn.tflite")

# 🔹 Float16 Quantization (lighter, keeps accuracy)
print("🔹 Converting to TFLite (Float16 quantized)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_fp16_model = converter.convert()
save_tflite_model(tflite_fp16_model, "char_recognizer_model_ffnn_fp16.tflite")

print("🎯 Conversion completed!")
print("✅ Generated models:")
print("- char_recognizer_model_ffnn.tflite (Float32)")
print("- char_recognizer_model_ffnn_fp16.tflite (Float16 Quantized)")
