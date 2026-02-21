import sys
sys.path.append('c:\\college\\SEM\\sem6\\DL[CSE-4006]\\Project_gesture')
import riva.client

API_KEY = 'Bearer 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'
FUNC_ID = '0778f2eb-b64d-45e7-acae-7dd9b9b35b4d' 

auth = riva.client.Auth(uri='grpc.nvcf.nvidia.com:443', use_ssl=True, metadata_args=[('function-id', FUNC_ID), ('authorization', API_KEY)])
service = riva.client.NeuralMachineTranslationClient(auth)

codes = {
    'English': 'en', 'Czech': 'cs', 'Danish': 'da', 'German': 'de', 'Greek': 'el',
    'European Spanish': 'es', 'LATAM Spanish': 'es', 'Finnish': 'fi', 'France': 'fr',
    'Hungarian': 'hu', 'Italian': 'it', 'Lithuanian': 'lt', 'Latvian': 'lv',
    'Dutch': 'nl', 'Norwegian': 'no', 'Polish': 'pl', 'European Portuguese': 'pt',
    'Brazillian Portuguese': 'pt', 'Romanian': 'ro', 'Russian': 'ru', 'Slovak': 'sk',
    'Swedish': 'sv', 'Simplified Chinese': 'zh', 'Traditional Chinese': 'zh',
    'Japanese': 'ja', 'Hindi': 'hi', 'Korean': 'ko', 'Estonian': 'et', 'Slovenian': 'sl',
    'Bulgarian': 'bg', 'Ukrainian': 'uk', 'Croatian': 'hr', 'Arabic': 'ar',
    'Vietnamese': 'vi', 'Turkish': 'tr', 'Indonesian': 'id', 'Thai': 'th'
}

passed = 0
failed = 0
for name, code in codes.items():
    if name == 'English': continue
    try:
         resp = service.translate(['Hello'], model='', source_language='en', target_language=code, future=False)
         print(f'[OK] {name} ({code}) -> {resp.translations[0].text}')
         passed += 1
    except Exception as e:
         print(f'[FAIL] {name} ({code}) -> Error: {str(e).split(chr(10))[0][:80]}')
         failed += 1

print(f'\nSummary: {passed} supported, {failed} unsupported/failed')
