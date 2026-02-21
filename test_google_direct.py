"""
Pinpoint test: mimics exactly what inference.py's translate_text does at runtime.
Run this while the server is STOPPED to test the translation pipeline.
"""
from deep_translator import GoogleTranslator

LANG_CODES = {
    "English": "en", "Arabic": "ar", "Bengali": "bn", "Bulgarian": "bg",
    "Simplified Chinese": "zh-CN", "Traditional Chinese": "zh-TW",
    "Croatian": "hr", "Czech": "cs", "Danish": "da", "Dutch": "nl",
    "Estonian": "et", "Finnish": "fi", "French": "fr", "German": "de",
    "Greek": "el", "Gujarati": "gu", "Hindi": "hi", "Hungarian": "hu",
    "Indonesian": "id", "Italian": "it", "Japanese": "ja", "Kannada": "kn",
    "Korean": "ko", "Latvian": "lv", "Lithuanian": "lt", "Malayalam": "ml",
    "Marathi": "mr", "Norwegian": "no", "Odia": "or", "Polish": "pl",
    "European Portuguese": "pt-PT", "Brazillian Portuguese": "pt-BR",
    "Punjabi": "pa", "Romanian": "ro", "Russian": "ru", "Slovak": "sk",
    "Slovenian": "sl", "European Spanish": "es-ES", "LATAM Spanish": "es-US",
    "Swedish": "sv", "Tamil": "ta", "Telugu": "te", "Thai": "th",
    "Turkish": "tr", "Ukrainian": "uk", "Urdu": "ur", "Vietnamese": "vi"
}

TEST_TEXT = "Hello this is Roshan."
INDIAN_LANGS = ["Bengali", "Gujarati", "Kannada", "Malayalam", "Marathi", "Odia", "Punjabi", "Tamil", "Telugu", "Urdu", "Hindi"]

print("=" * 60)
print("Direct GoogleTranslator test (same code as inference.py Step 2)")
print("=" * 60)

for lang in INDIAN_LANGS:
    tgt_code = LANG_CODES.get(lang)
    print(f"\n[{lang}] code={tgt_code}")
    try:
        result = GoogleTranslator(source="en", target=tgt_code).translate(TEST_TEXT)
        print(f"  OK: {result}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
