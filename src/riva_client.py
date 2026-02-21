import riva.client
import grpc
import os

class RivaTranslator:
    def __init__(self):
        # Configuration from User Snippet
        self.uri = "grpc.nvcf.nvidia.com:443"
        self.function_id = "0778f2eb-b64d-45e7-acae-7dd9b9b35b4d"
        self.api_key = "Bearer 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"
        
        try:
            self.auth = riva.client.Auth(
                uri=self.uri,
                use_ssl=True,
                metadata_args=[
                    ("function-id", self.function_id),
                    ("authorization", self.api_key)
                ]
            )
            self.service = riva.client.NeuralMachineTranslationClient(self.auth)
            print("SUCCESS: Connected to NVIDIA Riva NMT Service.")
        except Exception as e:
            print(f"ERROR: Failed to connect to Riva: {e}")
            self.service = None

        # Language Map (Project Language -> Riva ISO Code)
        # Verified from list-models: en, hi, es-ES, fr, de, etc.
        # Unknowns: Telugu, Tamil, etc. If not supported, will return None.
        self.lang_map = {
            "English": "en",
            "Arabic": "ar",
            "Bulgarian": "bg",
            "Simplified Chinese": "zh",
            "Traditional Chinese": "zh",
            "Croatian": "hr",
            "Czech": "cs",
            "Danish": "da",
            "Dutch": "nl",
            "Estonian": "et",
            "Finnish": "fi",
            "French": "fr",
            "German": "de",
            "Greek": "el",
            "Hindi": "hi",
            "Hungarian": "hu",
            "Indonesian": "id",
            "Italian": "it",
            "Japanese": "ja",
            "Korean": "ko",
            "Latvian": "lv",
            "Lithuanian": "lt",
            "Norwegian": "no",
            "Polish": "pl",
            "European Portuguese": "pt",
            "Brazillian Portuguese": "pt",
            "Romanian": "ro",
            "Russian": "ru",
            "Slovak": "sk",
            "Slovenian": "sl",
            "European Spanish": "es",
            "LATAM Spanish": "es",
            "Swedish": "sv",
            "Thai": "th",
            "Turkish": "tr",
            "Ukrainian": "uk",
            "Vietnamese": "vi",
            "Bengali": "bn",
            "Gujarati": "gu",
            "Kannada": "kn",
            "Malayalam": "ml",
            "Marathi": "mr",
            "Odia": "or",
            "Punjabi": "pa",
            "Tamil": "ta",
            "Telugu": "te",
            "Urdu": "ur"
        }

    def translate(self, text, target_language, source_language="English"):
        if not self.service:
            return self._google_translate(text, source_language, target_language)

        if target_language == source_language:
            return text

        src_code = self.lang_map.get(source_language)
        tgt_code = self.lang_map.get(target_language)

        if not src_code or not tgt_code:
            print(f"Riva: Unsupported language, routing to Google Translator: {source_language} -> {target_language}")
            return self._google_translate(text, source_language, target_language, src_code, tgt_code)

        # Try Riva NMT first (fastest, highest quality)
        try:
            response = self.service.translate(
                texts=[text],
                model="",
                source_language=src_code,
                target_language=tgt_code,
                future=False
            )
            if response.translations and response.translations[0].text:
                return response.translations[0].text
            # Riva returned empty response, fall through to Google
            print(f"Riva returned empty response for {target_language}. Falling back to Google Translator.")
        except Exception as e:
            print(f"Riva Error ({target_language}): {type(e).__name__} - Falling back to Google Translator.")

        # Always fall back to Google Translate if Riva fails or returns empty
        return self._google_translate(text, source_language, target_language, src_code, tgt_code)

    def _google_translate(self, text, source_language, target_language, src_code=None, tgt_code=None):
        """Fallback translator using free Google Translate via deep-translator."""
        print(f"Google Translator: {source_language} -> {target_language}")
        try:
            from deep_translator import GoogleTranslator

            if src_code is None:
                src_code = self.lang_map.get(source_language, "en")
            if tgt_code is None:
                tgt_code = self.lang_map.get(target_language)
            if not tgt_code:
                print(f"Google Translator: No code found for '{target_language}'")
                return None

            # Handle Google's specific dialect codes (zh-CN, zh-TW)
            gt_target = tgt_code
            if target_language == "Traditional Chinese": gt_target = "zh-TW"
            elif target_language == "Simplified Chinese": gt_target = "zh-CN"

            result = GoogleTranslator(source=src_code, target=gt_target).translate(text)
            return result
        except Exception as dt_e:
            print(f"Google Translator Error: {dt_e}")
            return None

class RivaTTS:
    def __init__(self):
        # Configuration (Assuming same credentials as NMT for now, or user to update)
        self.uri = "grpc.nvcf.nvidia.com:443"
        self.function_id = "0778f2eb-b64d-45e7-acae-7dd9b9b35b4d" # CHECK: Is this NMT-only or Full Riva?
        self.api_key = "Bearer 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"

        try:
            self.auth = riva.client.Auth(
                uri=self.uri,
                use_ssl=True,
                metadata_args=[
                    ("function-id", self.function_id),
                    ("authorization", self.api_key)
                ]
            )
            self.service = riva.client.SpeechSynthesisService(self.auth)
            print("SUCCESS: Connected to NVIDIA Riva TTS Service.")
        except Exception as e:
            print(f"ERROR: Failed to connect to Riva TTS: {e}")
            self.service = None

        # Voice Map (Language -> Voice Name)
        # Using standard Riva voice names (Generic fallback)
        self.voice_map = {
            "English": "English-US.Female-1",
            "Spanish": "Spanish-US.Female-1",
             # Add specific mappings if known, else rely on language_code defaults if possible
             # Riva requires explicit voice names usually.
        }
        
    def generate_audio_response(self, text, language_code):
        if not self.service:
            return None
            
        try:
            # Simple heuristic for voice name based on language code
            # Format: Language-Country.Gender-1
            # E.g., en-US.Female-1
            
            # Map simplified codes to Riva Voice prefixes
            # This is a BEST GUESS. Real deployment needs exact voice list.
            lang_prefix = "English-US"
            if language_code.startswith("es"): lang_prefix = "Spanish-US"
            elif language_code.startswith("de"): lang_prefix = "German-DE"
            elif language_code.startswith("fr"): lang_prefix = "French-FR"
            elif language_code.startswith("it"): lang_prefix = "Italian-IT"
            elif language_code.startswith("zh"): lang_prefix = "Chinese-CN"
            elif language_code.startswith("ja"): lang_prefix = "Japanese-JP"
            elif language_code.startswith("ru"): lang_prefix = "Russian-RU"
            elif language_code.startswith("hi"): lang_prefix = "Hindi-IN" # If supported
            
            voice_name = f"{lang_prefix}.Female-1"
            
            print(f"Riva TTS: Synthesizing '{text}' with voice '{voice_name}'")
            
            resp = self.service.synthesize(
                text,
                voice_name=voice_name,
                language_code=language_code,
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                sample_rate_hz=22050
            )
            
            # synthesize() returns a single SynthesizeSpeechResponse object
            audio_data = resp.audio
            return audio_data if audio_data else None

        except grpc.RpcError as e:
            # PERMISSION_DENIED  → this function-id doesn't cover TTS
            # UNIMPLEMENTED      → voice/language not available
            code = e.code().name if hasattr(e, 'code') else str(e)
            print(f"Riva TTS gRPC Error ({code}): {e.details() if hasattr(e, 'details') else e}")
            print("  ↳ Falling back to gTTS / offline engine.")
            return None
        except Exception as e:
            print(f"Riva TTS Error: {e}")
            return None
