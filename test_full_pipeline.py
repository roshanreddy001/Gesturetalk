import sys
from unittest.mock import MagicMock

# MOCK MEDIAPIPE BEFORE IMPORT
sys.modules["mediapipe"] = MagicMock()
sys.modules["mediapipe.solutions"] = MagicMock()
sys.modules["mediapipe.solutions.hands"] = MagicMock()
sys.modules["mediapipe.solutions.face_mesh"] = MagicMock()
sys.modules["mediapipe.solutions.drawing_utils"] = MagicMock()

# MOCK PYTTSX3
sys.modules["pyttsx3"] = MagicMock()

# MOCK TENSORFLOW
sys.modules["tensorflow"] = MagicMock()
sys.modules["tensorflow.keras"] = MagicMock()
sys.modules["tensorflow.keras.models"] = MagicMock()

# Now import inference
from src.inference import GestureRecognizer
import time

def test_pipeline():
    print("=== TESTING FULL PIPELINE ===")
    
    # 1. Initialize
    recognizer = GestureRecognizer()
    
    # 2. Check Enhancement State
    print(f"Initial Enhancement State: {recognizer.enhancement_enabled}")
    if not recognizer.enhancement_enabled:
        print("Enabling Enhancement...")
        recognizer.set_enhancement(True)
        
    # 3. Set Language
    target_lang = "Hindi"
    print(f"Setting Target Language to: {target_lang}")
    recognizer.set_language(target_lang)
    
    # 4. Simulate Input
    input_text = "how are you" # Lowercase as user might type
    print(f"Simulating Input: '{input_text}'")
    
    # Trigger (Async)
    recognizer.simulate_text(input_text)
    
    # Wait for Background Thread (2 seconds)
    print("Waiting for background processing...")
    time.sleep(3)
    
    # 5. Check Result
    final_text = recognizer.last_sentence
    print(f"Final Result: '{final_text}'")
    
    if final_text == input_text:
        print("FAILURE: Output equals Input (Translation Failed or not applied)")
    elif "The detailed answer" in final_text: 
         print("FAILURE: Enhanced text looks like Gemini gibberish?")
    else:
         print(f"SUCCESS: Translation occurred -> '{final_text}'")

if __name__ == "__main__":
    test_pipeline()
