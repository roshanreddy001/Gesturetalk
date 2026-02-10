import numpy as np
import math

class VideoPreprocessor:
    def __init__(self, sequence_length=10, num_features=84):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.buffer = [] # Stateful buffer for real-time inference
        
    def process_frame(self, landmarks, stateful=True):
        """
        Processes a single frame's landmarks.
        
        Args:
            landmarks: List of 42 (x, y) coordinates [x0, y0, x1, y1, ..., x20, y20] for ONE hand 
                       OR 84 coordinates for TWO hands.
            stateful: If True, appends to internal buffer and returns sequence if ready.
                      If False, just returns the processed feature vector.
                      
        Returns:
            If stateful=True: sequence (1, sequence_length, features) if buffer full, else None.
            If stateful=False: features (features,) array.
        """
        # 1. Normalize Landmarks (Wrist-relative + Scale-invariant)
        features = self._normalize_and_extract_features(landmarks)
        
        if not stateful:
            return features
            
        # 2. Update Buffer (Stateful Mode)
        self.buffer.append(features)
        
        # Keep only last N frames
        if len(self.buffer) > self.sequence_length:
            self.buffer.pop(0)
            
        # Return sequence if we have enough frames
        if len(self.buffer) == self.sequence_length:
            return np.expand_dims(np.array(self.buffer), axis=0)
            
        return None

    def reset(self):
        """Clears the temporal buffer."""
        self.buffer = []

    def _normalize_and_extract_features(self, landmarks):
        """
        Input: Raw List of coordinates (flat).
        Output: Normalized features + Angles/Distances.
        """
        # Handle 1 or 2 hands. Assumes 21 landmarks per hand * 2 coords = 42.
        # Structure: [Lx0, Ly0, ... Lx20, Ly20, Rx0, Ry0, ... Rx20, Ry20]
        # or just one hand (left or right padded with zeros?)
        # Based on inference.py, it sends left + right (84 items).
        
        data = np.array(landmarks, dtype=np.float32)
        
        # Split into Left and Right hand
        left_hand = data[0:42]
        right_hand = data[42:84]
        
        norm_left = self._process_single_hand(left_hand)
        norm_right = self._process_single_hand(right_hand)
        
        combined = np.concatenate([norm_left, norm_right])
        
        # Pad or truncate to fixed feature size if we add extra features later
        # For now, let's stick to the normalized coordinates (42+42 = 84)
        # TODO: Add angles/distances here if desired, improving num_features
        
        return combined

    def _process_single_hand(self, flat_hand):
        """
        Normalizes a single hand (42 points).
        Returns normalized 42 points.
        """
        # Check if hand is detected (not all zeros)
        if np.all(flat_hand == 0):
            return flat_hand
            
        # Reshape to (21, 2)
        points = flat_hand.reshape(-1, 2)
        
        # 1. Translation: Wrist (Index 0) becomes (0,0)
        wrist = points[0]
        points = points - wrist
        
        # 2. Scale: Distance between Wrist(0) and Middle Finger MCP(9)
        # This is a stable reference for hand size.
        # If distance is too small (noise), skip scaling
        dist = np.linalg.norm(points[9] - points[0])
        if dist > 0.01:
            points = points / dist
            
        return points.flatten()

    def create_sequences_from_data(self, X):
        """
        Helper for training: Converts a large array of frames into sequences.
        Assumes X is ordered (N, 84).
        Returns (N - seq_len + 1, seq_len, 84)
        """
        sequences = []
        # We need continuous data. If X comes from different sessions, 
        # this might bleed samples across sessions. 
        # For Phase 1 (synthesis), we'll assume continuous or handle carefully.
        
        for i in range(len(X) - self.sequence_length + 1):
            seq = X[i : i + self.sequence_length]
            sequences.append(seq)
            
        return np.array(sequences)
