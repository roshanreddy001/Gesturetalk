import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import os

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
CSV_FILE = os.path.join(DATA_DIR, 'keypoints.csv')
MODEL_PATH = os.path.join(MODELS_DIR, 'gesture_model.h5')
TFLITE_MODEL_PATH = os.path.join(MODELS_DIR, 'gesture_model.tflite')

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def load_data():
    print(f"DEBUG: Looking for data at: {os.path.abspath(CSV_FILE)}")
    if not os.path.exists(CSV_FILE):
        print(f"Error: Data file not found at {CSV_FILE}")
        return None, None
    
    df = pd.read_csv(CSV_FILE)
    X = df.iloc[:, :-1].values # All columns except last (features)
    y = df.iloc[:, -1].values  # Last column (labels)
    return X, y

def build_model(input_shape, num_classes):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_shape,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    print("Loading data...")
    X, y = load_data()
    if X is None:
        return

    # Check if we have enough data
    unique_classes = np.unique(y)
    print(f"DEBUG: Found classes: {unique_classes}")
    if len(unique_classes) < 2:
        print("Error: Need at least 2 gesture classes to train.")
        return

    print(f"Data shape: {X.shape}, Labels: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build model
    # Build model
    # Note: input_shape will be 42 (1 hand * 21 points * 2 coords)
    
    # Ensure output layer covers all potential classes (0-35)
    # If using sparse_categorical_crossentropy, we need neurons = max_label_index + 1
    num_classes = max(np.max(y) + 1, 34)
    
    model = build_model(X.shape[1], num_classes)
    model.summary()
    
    # Train
    print("Starting training...")
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    
    # Evaluate
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc:.4f}")
    
    # Save
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Convert to TFLite
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite Model saved to {TFLITE_MODEL_PATH}")

if __name__ == '__main__':
    main()
