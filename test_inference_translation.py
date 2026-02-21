import sys
sys.path.append('c:\\college\\SEM\\sem6\\DL[CSE-4006]\\Project_gesture')

from src.inference import GestureRecognizer

r = GestureRecognizer()
for lang in ['Bengali', 'Malayalam', 'Tamil', 'Telugu', 'Kannada']:
    r.set_language(lang)
    en, translated = r.translate_text('Hello this is Roshan.')
    print(f'{lang}: {translated}')

print('\nAll done.')
