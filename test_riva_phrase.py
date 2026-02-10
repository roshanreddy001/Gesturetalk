from src.riva_client import RivaTranslator

def test_phrase():
    print("=== TESTING SPECIFIC PHRASE ===")
    try:
        translator = RivaTranslator()
    except Exception as e:
        print(f"Failed to init Riva: {e}")
        return

    input_text = "How are you"
    target_lang = "Hindi" # 'hi'
    
    print(f"\nTranslating '{input_text}' -> {target_lang}...")
    
    try:
        # 1. Normal Call
        result = translator.translate(input_text, target_lang)
        print(f"Result: '{result}'")
        
        if result == input_text or not result:
             print("FAILURE: Returned original text or None.")
        else:
             print("SUCCESS: Translation seems to have happened.")
             
    except Exception as e:
         print(f"ERROR: {e}")

if __name__ == "__main__":
    test_phrase()
