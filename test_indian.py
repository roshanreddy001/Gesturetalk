import sys
sys.path.append('c:\\college\\SEM\\sem6\\DL[CSE-4006]\\Project_gesture')
from src.riva_client import RivaTranslator

t = RivaTranslator()
print("Translator ready.")

for lang in ['Malayalam', 'Tamil', 'Telugu', 'Kannada', 'Bengali']:
    print(f"\n--- Testing {lang} ---")
    try:
        result = t.translate('Hello this is Roshan.', lang)
        print(f'{lang}: {result}')
    except Exception as ex:
        print(f'{lang}: CRASHED with {type(ex).__name__}: {ex}')

print("\nAll done.")
