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

    def translate(self, text, target_language, source_language="English"):
        if not self.service:
            print("Riva Service not initialized.")
            return None

        if target_language == source_language:
            return text

        src_code = self.lang_map.get(source_language)
        tgt_code = self.lang_map.get(target_language)

        if not src_code or not tgt_code:
            print(f"Riva: Unsupported language pair {source_language} -> {target_language}")
            return None

        try:
            # print(f"DEBUG: Riva Translating '{text}' ({src_code}->{tgt_code})")
            response = self.service.translate(
                texts=[text],
                model="", # Default model
                source_language=src_code,
                target_language=tgt_code,
                future=False
            )
            
            if response.translations:
                translated_text = response.translations[0].text
                # print(f"DEBUG: Riva Result: {translated_text}")
                return translated_text
            return None

        except grpc.RpcError as e:
            # Catch specific errors (e.g. language not supported)
            # print(f"Riva RPC Error: {e.details()}")
            return None
        except Exception as e:
            print(f"Riva Error: {e}")
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
            self.service = riva.client.NeuralSpeechSynthesisClient(self.auth)
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
            
            responses = self.service.synthesize(
                text,
                voice_name=voice_name,
                language_code=language_code,
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                sample_rate_hz=22050
            )
            
            # Combine audio chunks
            audio_data = b""
            for resp in responses:
                audio_data += resp.audio
                
            return audio_data
            
        except Exception as e:
            print(f"Riva TTS Error: {e}")
            return None
