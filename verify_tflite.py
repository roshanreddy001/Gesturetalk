import cv2
import numpy as np
import time
from src.inference import GestureRecognizer

def verify():
    print("Initializing GestureRecognizer...")
    recognizer = GestureRecognizer()
    
    if recognizer.interpreter is None:
        print("FAILED: TFLite interpreter not loaded.")
        return
    
    print("SUCCESS: TFLite interpreter loaded.")
    
    # Create dummy frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    print("Processing dummy frame...")
    start_time = time.time()
    _, text, conf = recognizer.process_frame(dummy_frame)
    end_time = time.time()
    print(f"Frame processed in {end_time - start_time:.4f}s")
    print(f"Result: {text}")

    print("Testing Threaded Translation...")
    recognizer.enhancement_enabled = True
    recognizer.set_language("Hindi") # Test with a target language
    
    # Simulate an event
    result = recognizer.process_sentence_event("Hello")
    print(f"Immediate Result (Should be raw 'Hello'): {result}")
    
    # Wait for thread (mocking wait)
    print("Waiting for background thread...")
    time.sleep(2) 
    
    print(f"Final Last Sentence: {recognizer.last_sentence}")
    print(f"Update ID: {recognizer.last_update_id}")
    
    print("Verification Completed.")

if __name__ == "__main__":
    verify()
