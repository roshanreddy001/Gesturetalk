import tensorflow as tf
import os

# Define paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'gesture_model.keras')
TFLITE_MODEL_PATH = os.path.join(MODELS_DIR, 'gesture_model.tflite')

def convert_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        
        # Convert the model
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Optional: Optimize for size/speed (Default optimization)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()

        # Save the model
        with open(TFLITE_MODEL_PATH, 'wb') as f:
            f.write(tflite_model)
            
        print(f"Success! Model converted and saved to {TFLITE_MODEL_PATH}")
        
    except Exception as e:
        print(f"Conversion failed: {e}")

if __name__ == "__main__":
    convert_model()
