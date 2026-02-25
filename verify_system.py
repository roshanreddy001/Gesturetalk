import traceback
import sys
import os

print("--- System Verification ---")

try:
    print("1. Testing Imports...")
    from src.inference import GestureRecognizer, GESTURES
    from src.data_collection import GESTURES as DATA_GESTURES
    from src.offline_dict import OFFLINE_TRANSLATIONS, get_offline_translation
    print("   [+] Imports successful.")

    print("\n2. Testing Constants Consistency...")
    assert len(GESTURES) == 15, "Inference GESTURES should be 15."
    assert len(DATA_GESTURES) == 15, "Data collection GESTURES should be 15."
    assert GESTURES == DATA_GESTURES, "Gesture dictionaries mismatch between inference and data collection."
    print("   [+] Constants are consistent.")

    print("\n3. Testing Inference Initialization...")
    recognizer = GestureRecognizer()
    print("   [+] GestureRecognizer initialized successfully.")

    print("\n4. Testing GestureBuffer Dataset Integration...")
    # Buffer should have loaded the CSV
    assert len(recognizer.gesture_buffer.sequence_map) > 0, "Sequence map not loaded from CSV"
    
    # Test a combination that exists in the template
    recognizer.gesture_buffer.buffer = ["Hello", "Please", "Help"]
    final = recognizer.gesture_buffer.finalize_sentence()
    print(f"   [+] Combined 'Hello+Please+Help' -> '{final}'")
    assert final == "Hello, can you please help me?", f"Unexpected sentence: {final}"
    
    # Test fallback
    recognizer.gesture_buffer.buffer = ["Water", "Home", "Wait", "Stop"]
    final_fallback = recognizer.gesture_buffer.finalize_sentence()
    print(f"   [+] Fallback 'Water+Home+Wait+Stop' -> '{final_fallback}'")
    
    print("\n5. Testing Translation Fallback...")
    # Should fallback offline or English depending on language
    eng_text, tr_text = recognizer.translate_text("Water")
    print(f"   [+] English to English: {tr_text}")
    
    recognizer.set_language("Tamil")
    eng_text, tr_text = recognizer.translate_text("Water")
    print(f"   [+] English to Tamil (Water): {tr_text}")

    print("\n--- All Checks Passed Successfully! ---")
    sys.exit(0)

except AssertionError as e:
    print(f"\n[!] ASSERTION FAILED: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n[!] ERROR ENCOUNTERED: {e}")
    traceback.print_exc()
    sys.exit(1)
