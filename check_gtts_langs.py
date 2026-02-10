from gtts.lang import tts_langs
try:
    langs = tts_langs()
    for code, name in langs.items():
        print(f"{code}: {name}")
except Exception as e:
    print(f"Error: {e}")
