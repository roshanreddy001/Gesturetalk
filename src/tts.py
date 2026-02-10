try:
    from gtts import gTTS
except ImportError:
    gTTS = None
    print("Warning: gTTS not found. Online speech disabled.")

import io
import base64
import os
import wave
import threading

# Global lock for TTS engine (pyttsx3 is not thread-safe)
tts_lock = threading.Lock()

try:
    from src.riva_client import RivaTTS
    riva_tts = RivaTTS()
except Exception as e:
    print(f"Riva TTS Import Failed: {e}")
    riva_tts = None

# Language Mapping from Frontend names to gTTS codes
# Language Mapping (Frontend Name -> Standard ISO Code)
LANG_MAP = {
    "English": "en",
    "Arabic": "ar",
    "Bengali": "bn",
    "Bulgarian": "bg",
    "Simplified Chinese": "zh-CN",
    "Traditional Chinese": "zh-TW",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "Estonian": "et",
    "Finnish": "fi",
    "French": "fr",
    "German": "de",
    "Greek": "el",
    "Gujarati": "gu",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Indonesian": "id",
    "Italian": "it",
    "Japanese": "ja",
    "Kannada": "kn",
    "Korean": "ko",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Norwegian": "no",
    "Odia": "or",
    "Polish": "pl",
    "European Portuguese": "pt-PT",
    "Brazillian Portuguese": "pt-BR",
    "Punjabi": "pa",
    "Romanian": "ro",
    "Russian": "ru",
    "Slovak": "sk",
    "Slovenian": "sl",
    "European Spanish": "es-ES",
    "LATAM Spanish": "es-US",
    "Swedish": "sv",
    "Tamil": "ta",
    "Telugu": "te",
    "Thai": "th",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Vietnamese": "vi"
}

def generate_audio(text, language_name):
    """
    Generates audio for the given text and language.
    Returns: Base64 encoded MP3 string.
    """
    if riva_tts:
        try:
            # Riva uses specific codes (e.g., en-US). 
            # Our Map has basic ISO or full codes.
            lang_code = LANG_MAP.get(language_name, "en-US")
            
            # Helper to map generic codes to Riva expected format if needed
            if lang_code == "en": lang_code = "en-US"
            
            audio_raw = riva_tts.generate_audio_response(text, lang_code)
            if audio_raw:
                # Riva returns raw PCM/WAV bytes? 
                # NeuralSpeechSynthesisClient usually returns Linear PCM.
                # We need to wrap it in WAV container for browser to play easily as base64
                if len(audio_raw) > 0:
                     # Create WAV in memory
                    wav_io = io.BytesIO()
                    with wave.open(wav_io, "wb") as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2) # 16-bit PCM
                        wav_file.setframerate(22050) # Assuming 22050Hz default
                        wav_file.writeframes(audio_raw)
                    
                    wav_io.seek(0)
                    return base64.b64encode(wav_io.read()).decode('utf-8')
        except Exception as e:
            print(f"Riva TTS Attempt Failed: {e}")

    if gTTS:
        try:
            lang_code = LANG_MAP.get(language_name, "en")
            
            try:
                # Create gTTS object
                tts = gTTS(text=text, lang=lang_code, slow=False)
                mp3_fp = io.BytesIO()
                tts.write_to_fp(mp3_fp)
                
                # Encode to base64
                mp3_fp.seek(0)
                audio_b64 = base64.b64encode(mp3_fp.read()).decode('utf-8')
                return audio_b64
            except Exception as e:
                print(f"TTS Error ({lang_code}): {e}. Attempting Offline Fallback.")
                # Fallback to Offline TTS
                return generate_offline_audio(text)
        except Exception as e:
            print(f"Critical TTS Error: {e}")
            return None
    else:
        # gTTS missing, force offline
        return generate_offline_audio(text)

# Initialize offline engine once
try:
    import pyttsx3
    offline_engine = pyttsx3.init()
    offline_engine.setProperty('rate', 150) # Speed
except:
    offline_engine = None

def generate_offline_audio(text):
    """
    Fallback to pyttsx3 (Offline)
    """
    if not offline_engine:
        return None
    
    try:
        # Thread-safe access to engine
        with tts_lock:
            # Save to temp file
            temp_file = "temp_speech.mp3"
            offline_engine.save_to_file(text, temp_file)
            offline_engine.runAndWait()
        
        # Read bytes
        with open(temp_file, "rb") as f:
            audio_data = f.read()
            
        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        return base64.b64encode(audio_data).decode('utf-8')
    except Exception as e:
        print(f"Offline TTS Error: {e}")
        return None
