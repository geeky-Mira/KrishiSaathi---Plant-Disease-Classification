# quantize.py
import tensorflow as tf

# 1. Load your saved Keras model
model = tf.keras.models.load_model("plant-disease-detection-model.keras")

# 2. Convert with default post‑training quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # 8‑bit weight quantization :contentReference[oaicite:5]{index=5}
tflite_quant = converter.convert()

# 3. Save the quantized TFLite model
with open("plant_disease_model_quantized.tflite", "wb") as f:
    f.write(tflite_quant)

print("✅ Quantized model saved to plant_disease_model_quantized.tflite")
