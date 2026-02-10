from src.offline_dict import get_offline_translation
from src.tts import LANG_MAP
import sys

def test_language_transition():
    # Updated to test gesture KEYS, not phrases
    test_phrases = [
        "Hello", "Thank You", "No", "Please", "Emergency",
        "Washroom", "Call Doctor", "Sick", "Home", ""
    ]
    
    # Target Languages to test (Indic + European)
    target_languages = [
        "English", 
        "Hindi", 
        "Telugu", 
        "Tamil", 
        "Spanish", 
        "French",
        "Urdu",
        "Odia" # Check missing vs available
    ]
    
    print("=== LANGUAGE TRANSITION TEST ===")
    
    start_failures = 0
    
    for lang in target_languages:
        print(f"\n--- Testing Language: {lang} ---")
        
        # 1. Check TTS Mapping
        tts_code = LANG_MAP.get(lang)
        if tts_code:
            print(f"TTS Engine Code: '{tts_code}' (OK)")
        else:
            print(f"WARNING: No TTS mapping for '{lang}'. Falls back to English?")
            
        # 2. Check Translations
        print("Translations:")
        
        # Check support
        is_supported = get_offline_translation("Hello", lang) != "Hello"
        if lang == "English": is_supported = True
        
        if not is_supported:
             print(f"  [INFO] Language '{lang}' not in offline dict. Expecting English.")
        
        for phrase in test_phrases:
            if not phrase: continue
            
            # Test direct lookup
            trans = get_offline_translation(phrase, lang)
            
            # Test punctuation handling
            punctuated_phrase = phrase + "."
            trans_punct = get_offline_translation(punctuated_phrase, lang)
            
            status = "OK"
            if trans == phrase and lang != "English" and is_supported:
                status = "MISSING (Returns English)"
            
            # Only fail punctuation check if language is supported AND distinct from English
            if trans != trans_punct and lang != "English":
                status = "FAIL (Punctuation Logic Broken)"
                start_failures += 1
                
            print(f"  '{phrase}' -> '{trans}' [{status}]")
            
    if start_failures == 0:
        print("\nSUCCESS: All punctuation checks passed. Language transitions smooth.")
    else:
        print(f"\nFAILURE: {start_failures} punctuation handling errors.")

if __name__ == "__main__":
    test_language_transition()
