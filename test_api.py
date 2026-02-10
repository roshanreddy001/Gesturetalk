from src.inference import GestureRecognizer 
import time
import sys

def test_gemini_api():
    print("=== GEMINI API INTEGRATION TEST ===")
    
    # Initialize the Recognizer
    try:
        recognizer = GestureRecognizer()
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize GestureRecognizer: {e}")
        return

    # Check if Gemini Model is active
    if not recognizer.gemini_model:
        print("FAILURE: Gemini Model is NOT initialized. Check API Key or availability.")
        return
    else:
        print("SUCCESS: Gemini Model initialized.")
        # Access internal model name if possible, or just print object
        # print(f"Using Model: {recognizer.gemini_model.model_name}") # Hypothetical

    # Enable Enhancement
    recognizer.set_enhancement(True)
    
    # Test Case 1: English -> Spanish
    input_text = "Hello"
    target_lang = "Spanish"
    print(f"\n--- Test 1: '{input_text}' -> '{target_lang}' ---")
    recognizer.set_language(target_lang)
    
    start_time = time.time()
    refined_english, translated = recognizer.translate_text(input_text)
    duration = time.time() - start_time
    
    print(f"Time Taken: {duration:.2f}s")
    print(f"Refined English: '{refined_english}'")
    print(f"Translated: '{translated}'")
    
    if refined_english == input_text and translated == input_text:
        print("RESULT: FAIL (Output identical to input. Fallback likely used.)")
    else:
        print("RESULT: PASS (Enhanced output received)")

    # Test Case 2: English -> Hindi
    input_text = "Hungry"
    target_lang = "Hindi"
    print(f"\n--- Test 2: '{input_text}' -> '{target_lang}' ---")
    recognizer.set_language(target_lang)
    
    start_time = time.time()
    refined_english, translated = recognizer.translate_text(input_text)
    duration = time.time() - start_time
    
    print(f"Time Taken: {duration:.2f}s")
    print(f"Refined English: '{refined_english}'")
    print(f"Translated: '{translated}'")
    
    if refined_english == input_text and translated == input_text:
        print("RESULT: FAIL (Output identical to input. Fallback likely used.)")
    else:
        print("RESULT: PASS (Enhanced output received)")

if __name__ == "__main__":
    test_gemini_api()
