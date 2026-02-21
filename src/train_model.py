import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import os

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

X_FILE = os.path.join(DATA_DIR, 'X_sequences.npy')
Y_FILE = os.path.join(DATA_DIR, 'y_labels.npy')
MODEL_PATH = os.path.join(MODEL_DIR, 'gesture_model.keras')
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, 'gesture_model.tflite')

def load_data():
    if not os.path.exists(X_FILE) or not os.path.exists(Y_FILE):
        print("Data files not found. Run synthesize_dataset.py first.")
        return None, None, None
        
    X = np.load(X_FILE)
    y = np.load(Y_FILE)
    
    # Num classes = max label + 1 (assuming 0-indexed)
    num_classes = int(np.max(y) + 1)
    
    return X, y, num_classes

def create_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        
        # Conv1D Layer 1: Spatial + Temporal features
        Conv1D(128, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.2), # Lightweight dropout
        
        # Conv1D Layer 2
        Conv1D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.2),
        
        # Global Pooling to reduce parameters and handle variable length (if needed)
        GlobalAveragePooling1D(),
        
        # Dense Output
        Dense(64, activation='relu', kernel_initializer='he_normal'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    return model

def representative_data_gen():
    """Generates data for INT8 quantization."""
    # Load a small subset of data for calibration
    # Ideally should use validation set, but loading everything here for simplicity
    X = np.load(X_FILE)
    # Take first 100 samples
    for i in range(min(100, len(X))):
        input_data = X[i].astype(np.float32)
        input_data = np.expand_dims(input_data, axis=0) # Add batch dim
        yield [input_data]

def train_and_convert():
    print("Loading data...")
    X, y, num_classes = load_data()
    if X is None: return

    print(f"Data shape: {X.shape}, Labels: {y.shape}, Classes: {num_classes}")
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create Model
    # Input shape: (TimeSteps, Features) -> (10, 84)
    model = create_model(input_shape=(X.shape[1], X.shape[2]), num_classes=num_classes)
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
    
    # Train
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr]
    )
    
    # Save Keras Model
    print(f"Saving model to {MODEL_PATH}")
    model.save(MODEL_PATH)
    
    # TFLite Conversion
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Representative Dataset for INT8 Quantization
    # This requires the model inputs to be Float32, but weights to be INT8
    # For full integer quantization, we need more config, but 'DEFAULT' + rep dataset 
    # gives dynamic range quantization + some int8 weights.
    # To enforce full INT8 input/output, we'd need:
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.inference_input_type = tf.int8
    # converter.inference_output_type = tf.int8
    # But let's stick to hybrid/dynamic first for compatibility.
    
    converter.representative_dataset = representative_data_gen
    
    try:
        tflite_model = converter.convert()
        
        with open(TFLITE_MODEL_PATH, 'wb') as f:
            f.write(tflite_model)
        print(f"Saved TFLite model to {TFLITE_MODEL_PATH}")
        print(f"TFLite Model Size: {len(tflite_model) / 1024:.2f} KB")
        
    except Exception as e:
        print(f"Error converting to TFLite: {e}")

if __name__ == "__main__":
    train_and_convert()
