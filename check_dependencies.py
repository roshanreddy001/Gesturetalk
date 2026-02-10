import sys
print(f"Python: {sys.version}")

try:
    import tensorflow as tf
    print(f"TensorFlow: {tf.__version__}")
except ImportError as e:
    print(f"TensorFlow Import Error: {e}")

try:
    import cv2
    print(f"OpenCV: {cv2.__version__}")
except ImportError as e:
    print(f"OpenCV Import Error: {e}")

try:
    import mediapipe as mp
    print(f"MediaPipe: {mp.__version__}")
except ImportError as e:
    print(f"MediaPipe Import Error: {e}")

try:
    import pyttsx3
    print("pyttsx3: Imported")
except ImportError as e:
    print(f"pyttsx3 Import Error: {e}")
