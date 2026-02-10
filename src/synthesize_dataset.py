import sys
import os

# Fix path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from src.video_preprocessor import VideoPreprocessor

try:
    import cv2
    print(f"OpenCV Version: {cv2.__version__}")
except ImportError as e:
    print(f"Warning: OpenCV import failed: {e}")

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
INPUT_CSV = os.path.join(DATA_DIR, 'keypoints.csv')
OUTPUT_X = os.path.join(DATA_DIR, 'X_sequences.npy')
OUTPUT_Y = os.path.join(DATA_DIR, 'y_labels.npy')

SEQUENCE_LENGTH = 10

def synthesize_data():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    print(f"Loading data from {INPUT_CSV}...")
    
    # Try reading with header first to see if 'label' exists
    try:
        df = pd.read_csv(INPUT_CSV)
        if 'label' in df.columns:
            print("Header detected.")
            X_raw = df.iloc[:, :-1].values
            y_raw = df['label'].values
        else:
            raise ValueError("Label column not found")
    except Exception:
        print("Header not found or invalid. Assuming raw data (85 cols).")
        # Reload without header
        df = pd.read_csv(INPUT_CSV, header=None)
        # Check if we have 85 columns (84 features + 1 label)
        if df.shape[1] == 85:
            X_raw = df.iloc[:, :-1].values
            y_raw = df.iloc[:, -1].values
        else:
            print(f"Error: Unexpected column count {df.shape[1]}. Expected 85.")
            return

    print(f"Raw Data Shape: {X_raw.shape}")
    
    preprocessor = VideoPreprocessor(sequence_length=SEQUENCE_LENGTH)
    
    # 1. Normalize All Frames
    print("Normalizing frames...")
    X_norm = []
    for i in range(len(X_raw)):
        # Reshape to (84,) if needed, though it's already flat
        frame = X_raw[i]
        # Our preprocessor expects flat 84-vector
        norm_frame = preprocessor.process_frame(frame, stateful=False)
        X_norm.append(norm_frame)
        
    X_norm = np.array(X_norm)
    
    # 2. Synthesize Sequences
    # We assume data is grouped by label. 
    # We MUST NOT bleed sequences across different labels.
    
    unique_labels = np.unique(y_raw)
    final_sequences = []
    final_labels = []
    
    print(f"Synthesizing sequences for {len(unique_labels)} classes...")
    
    for label in unique_labels:
        # Get all frames for this label
        indices = np.where(y_raw == label)[0]
        
        # If we have discontinuous blocks of the same label, this simple approach 
        # might bridge the gap. But usually data collection is one session per label.
        # Ideally we'd have session IDs. 
        # For Phase 1, we assume continuous blocks per label.
        
        class_data = X_norm[indices]
        
        if len(class_data) < SEQUENCE_LENGTH:
            print(f"Warning: Class {label} has too few samples ({len(class_data)}). Skipping.")
            continue
            
        # Create sequences
        cls_sequences = preprocessor.create_sequences_from_data(class_data)
        
        # Create matching labels
        cls_labels = [label] * len(cls_sequences)
        
        final_sequences.extend(cls_sequences)
        final_labels.extend(cls_labels)
        
    final_sequences = np.array(final_sequences)
    final_labels = np.array(final_labels)
    
    print(f"Generated Sequences: {final_sequences.shape}")
    print(f"Generated Labels: {final_labels.shape}")
    
    # Save
    np.save(OUTPUT_X, final_sequences)
    np.save(OUTPUT_Y, final_labels)
    print(f"Saved to {OUTPUT_X} and {OUTPUT_Y}")

if __name__ == "__main__":
    synthesize_data()
