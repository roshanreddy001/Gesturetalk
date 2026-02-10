from src.riva_client import RivaTranslator

def test_riva_langs():
    print("=== TESTING RIVA LANGUAGES ===")
    try:
        translator = RivaTranslator()
    except Exception as e:
        print(f"Failed to init Riva: {e}")
        return

    # Languages from our project
    test_langs = [
        "Hindi", "Telugu", "Tamil", "Kannada", "Malayalam", 
        "Marathi", "Bengali", "Gujarati", "Odia", 
        "Spanish", "French", "German", "Urdu"
    ]
    
    input_text = "Hello"
    
    for lang in test_langs:
        print(f"\nTesting English -> {lang}...")
        try:
            result = translator.translate(input_text, lang)
            if result:
                print(f"  [SUCCESS] {lang}: '{result}'")
            else:
                print(f"  [FAILED] {lang}: Returned None (Unsupported?)")
        except Exception as e:
             print(f"  [ERROR] {lang}: {e}")

if __name__ == "__main__":
    test_riva_langs()
