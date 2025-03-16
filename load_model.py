import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import tflite_runtime.interpreter as tflite  # ✅ Use tflite-runtime!

# ✅ Load the TFLite FP16 model
model_path = "char_recognizer_model_ffnn_fp16.tflite"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ TFLite model not found: {model_path}")

print(f"✅ Loading TFLite model from {model_path}...")
interpreter = tflite.Interpreter(model_path=model_path)  # Use tflite runtime interpreter
interpreter.allocate_tensors()
print("✅ TFLite model loaded and allocated successfully!")

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# ✅ Function to predict digit from image array
def predict_digit(image_array):
    """
    Receives a (1, 28, 28) preprocessed grayscale image array.
    Returns the predicted digit as a string.
    """

    # Ensure correct type and shape
    image_array = image_array.astype(np.float32)  # TFLite expects float32 even for FP16

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], image_array)

    # Run inference
    interpreter.invoke()

    # Get output tensor (probabilities for digits 0-9)
    output = interpreter.get_tensor(output_details[0]['index'])

    # Get digit with highest probability
    predicted_digit = np.argmax(output)

    return str(predicted_digit)
