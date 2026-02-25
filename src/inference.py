import cv2
import mediapipe as mp
import numpy as np
import tflite_runtime.interpreter as tf
import os
import pyttsx3
import threading
from enum import Enum
import time
import csv
from src.offline_dict import get_offline_translation
from src.riva_client import RivaTranslator
from src.text_processor import TextProcessor
from src.video_preprocessor import VideoPreprocessor
from collections import deque, Counter

# Constants
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
# We now use the TFLite model from the data directory (or models dir if moved)
# train_model.py saves to data/gesture_model.tflite usually, let's update this to match
# Actually, train_model.py had TFLite path as .../data/gesture_model.tflite but updated 
# version saves to MODELS_DIR? Let's check train_model.py output logic.
# It saves to MODELS_DIR/gesture_model.tflite.
MODEL_PATH = os.path.join(MODELS_DIR, 'gesture_model.tflite')

# Define gestures mapping (Must match data collection)
GESTURES = {
    0: "Hello",
    1: "Goodbye",
    2: "Yes",
    3: "No",
    4: "Please",
    5: "Thank You",
    6: "Sorry",
    7: "Help",
    8: "Stop",
    9: "Wait",
    10: "Hungry",
    11: "Water",
    12: "Pain",
    13: "Emergency",
    14: "Home"
}

# Map Gestures to Full Sentences (for Speech)
PHRASE_MAP = {
    "Hello": "Hello there",
    "Goodbye": "Goodbye, see you later",
    "Yes": "Yes, I agree",
    "No": "No, I disagree",
    "Please": "Please",
    "Thank You": "Thank you very much",
    "Sorry": "I am sorry",
    "Help": "Please help me",
    "Stop": "Stop",
    "Wait": "Please wait",
    "Hungry": "I am hungry",
    "Water": "I need water",
    "Pain": "I am in pain",
    "Emergency": "This is an emergency",
    "Home": "I want to go home"
}

# --- Multimodal Context Helpers ---

def get_facial_emotion(face_landmarks):
    """
    Heuristic-based emotion detection from facial landmarks.
    Returns: "Happy", "Sad", "Angry", "Scared", or "Neutral"
    """
    if not face_landmarks:
        return "Neutral"

    # Shortcuts for landmarks
    lm = face_landmarks.landmark
    
    # Smile Detection (Mouth Width vs Height)
    # Left Corner: 61, Right Corner: 291, Top Lip: 13, Bottom Lip: 14
    mouth_width = abs(lm[61].x - lm[291].x)
    mouth_height = abs(lm[13].y - lm[14].y)
    smile_ratio = mouth_width / (mouth_height + 0.001)

    if smile_ratio > 4.5: 
        return "Happy"

    # Frown/Sad Detection (Corners lower than center?)
    corner_avg_y = (lm[61].y + lm[291].y) / 2
    center_avg_y = (lm[13].y + lm[14].y) / 2
    if corner_avg_y > center_avg_y + 0.02: 
        return "Sad"

    # Surprised/Scared (Eyes wide open)
    # Left Eye: 159-145, Right Eye: 386-374
    left_eye_open = abs(lm[159].y - lm[145].y)
    right_eye_open = abs(lm[386].y - lm[374].y)
    if left_eye_open > 0.04 and right_eye_open > 0.04: 
        return "Scared"

    return "Neutral"

class GestureBuffer:
    def __init__(self):
        self.buffer = []
        self.last_gesture = None
        self.phrase_map = PHRASE_MAP
        self.sequence_map = self._load_sequence_map()

    def _load_sequence_map(self):
        """Loads the pre-generated dataset mapping combinations of gestures to natural sentences."""
        mapping = {}
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'gesture_sentences.csv')
        try:
            if os.path.exists(csv_path):
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader) # skip header
                    for row in reader:
                        if len(row) >= 2:
                            mapping[row[0]] = row[1]
                print(f"Loaded {len(mapping)} gesture sequences from dataset.")
            else:
                print("Sequence map dataset not found. Using fallback join.")
        except Exception as e:
            print(f"Error loading sequence map: {e}")
            
        return mapping

    def add_gesture(self, gesture, emotion="Neutral", hand_count=1):
        # Trigger Sentence Finalization
        if gesture in ["Stop", "Silence"]:
             return self.finalize_sentence()
        
        # Avoid duplicate consecutive gestures
        if gesture != self.last_gesture:
            # Use RAW gesture name (Gemini will refine it later)
            phrase = gesture
            
            self.buffer.append(phrase)
            self.last_gesture = gesture
            return None # Buffering (no immediate speech)
            
        return None

    def finalize_sentence(self):
        if not self.buffer:
            return None
            
        # 1. Try exact sequence mapping
        sequence_key = "+".join(self.buffer)
        
        # 2. Check if there's a predefined natural translation for this pattern
        if sequence_key in self.sequence_map:
            sentence = self.sequence_map[sequence_key]
        else:
            # Fallback: Use exact phrase map
            phrases = [self.phrase_map.get(g, g) for g in self.buffer]
            if len(phrases) == 1:
                sentence = phrases[0]
            else:
                sentence = ". ".join(phrases) + "."
            
        self.buffer = []
        self.last_gesture = None
        return sentence

    def get_current_buffer(self):
        return " + ".join(self.buffer)


class GestureRecognizer:
    def __init__(self):
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        
        # Load TFLite Model
        if os.path.exists(MODEL_PATH):
            try:
                self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"TFLite Model loaded from {MODEL_PATH}")
            except Exception as e:
                print(f"Error loading TFLite model: {e}")
                self.interpreter = None
        else:
            print(f"Warning: Model not found at {MODEL_PATH}")

        # Components
        self.preprocessor = VideoPreprocessor(sequence_length=10) # Stateful buffer
        # Increase maxlen to increase the time needed to "hold" a gesture (e.g., 20 frames = ~0.6 seconds at 30fps)
        self.prediction_buffer = deque(maxlen=30) # Smoothing buffer
        
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=0  # 0=Lite (Faster), 1=Full (Default)
        )
        
        self.prediction_history = []
        self.history_length = 5 # Reduced from 8 for faster response (~150ms @ 30fps)
        self.gesture_buffer = GestureBuffer()
        
        # State Flags
        self.speech_enabled = True
        self.mode = "live"
        self.last_sentence = "" # Stores the final output (Translated)
        self.last_english_sentence = "" # Stores the English base
        self.last_audio = None # Stores the generated audio (base64)
        self.last_update_id = 0 # Unique ID for each finalized sentence event
        
        # Hands-off Trigger State
        self.last_hands_visible_time = 0
        self.HANDS_OFF_COOLDOWN = 0.2 # Seconds
        self.sentence_triggered = False

        # Optimization: Frame Skipping
        self.frame_count = 0
        self.skip_frames = 0 # Process every frame for smooth video
        self.last_predicted_text = ""
        self.last_confidence = 0.0

        # NVIDIA Riva Translation (Primary)
        self.riva_translator = None
        try:
             self.riva_translator = RivaTranslator()
        except Exception as e:
             print(f"Warning: Failed to initialize Riva: {e}")
             self.riva_translator = None

        self.target_language = "English"

        # Text Processor (Grammar & Custom Rules)
        self.text_processor = TextProcessor()
        
        # Thread Safety
        self.lock = threading.Lock()

    def set_language(self, language):
        self.target_language = language
        print(f"Target Language set to: {self.target_language}")

    # Languages that Riva does NOT support – go directly to Google Translate
    RIVA_UNSUPPORTED = {
        "Bengali", "Gujarati", "Kannada", "Malayalam", "Marathi",
        "Odia", "Punjabi", "Tamil", "Telugu", "Urdu"
    }

    # Google Translate language codes (used in both fallback paths)
    GOOGLE_LANG_CODES = {
        "English": "en", "Arabic": "ar", "Bengali": "bn", "Bulgarian": "bg",
        "Simplified Chinese": "zh-CN", "Traditional Chinese": "zh-TW",
        "Croatian": "hr", "Czech": "cs", "Danish": "da", "Dutch": "nl",
        "Estonian": "et", "Finnish": "fi", "French": "fr", "German": "de",
        "Greek": "el", "Gujarati": "gu", "Hindi": "hi", "Hungarian": "hu",
        "Indonesian": "id", "Italian": "it", "Japanese": "ja", "Kannada": "kn",
        "Korean": "ko", "Latvian": "lv", "Lithuanian": "lt", "Malayalam": "ml",
        "Marathi": "mr", "Norwegian": "no", "Odia": "or", "Polish": "pl",
        "European Portuguese": "pt", "Brazillian Portuguese": "pt",
        "Punjabi": "pa", "Romanian": "ro", "Russian": "ru", "Slovak": "sk",
        "Slovenian": "sl", "European Spanish": "es", "LATAM Spanish": "es",
        "Swedish": "sv", "Tamil": "ta", "Telugu": "te", "Thai": "th",
        "Turkish": "tr", "Ukrainian": "uk", "Urdu": "ur", "Vietnamese": "vi"
    }

    def translate_text(self, text):
        """
        Returns a tuple: (English Text, Translated Text)
        Uses Riva NMT for supported languages, Google Translate for the rest.
        """
        print(f"\n=== TRANSLATE: '{text}' → [{self.target_language}] ===")

        if self.target_language == "English":
            print("  PATH: English → no translation needed")
            return text, text

        # FAST PATH: Known Riva-unsupported languages → skip Riva, use Google directly
        if self.target_language in GestureRecognizer.RIVA_UNSUPPORTED:
            print(f"  PATH: FAST → Google Translate for {self.target_language}")
            result = self._google_translate_direct(text, self.target_language)
            if result:
                return text, result
            # Fall through to offline dict if even Google fails
            print("  PATH: Google failed → offline dict")
            return text, get_offline_translation(text, self.target_language)

        print(f"  PATH: Riva NMT for {self.target_language}")
        # STEP 1: Attempt Riva NMT (Primary - fastest, highest quality)
        if self.riva_translator:
            try:
                translated_text = self.riva_translator.translate(text, self.target_language)
                if translated_text and translated_text.strip() and translated_text != text:
                    print(f"### RIVA SUCCESS ### {self.target_language}: {translated_text}")
                    return text, translated_text
                else:
                    print(f"DEBUG: Riva returned unchanged text. Using Google Translate fallback.")
            except Exception as e:
                print(f"!!! RIVA FAILED !!! Error: {e}")

        # STEP 2: Google Translate fallback for Riva-supported languages that failed
        result = self._google_translate_direct(text, self.target_language)
        if result:
            return text, result

        # STEP 3: Offline Fallback (last resort)
        translated = get_offline_translation(text, self.target_language)
        return text, translated

    def _google_translate_direct(self, text, target_language):
        """Translate directly via Google Translate (deep-translator). Returns None on failure."""
        tgt_code = GestureRecognizer.GOOGLE_LANG_CODES.get(target_language)
        print(f"  Google Translate: lang={target_language}, code={tgt_code}")
        if not tgt_code:
            print(f"  Google Translate: ERROR - No code found for '{target_language}'")
            return None
        try:
            from deep_translator import GoogleTranslator
            print(f"  Google Translate: calling API... '{text}' → {tgt_code}")
            result = GoogleTranslator(source="en", target=tgt_code).translate(text)
            print(f"  Google Translate: API result = '{result}'")
            if result and result.strip():
                print(f"  Google Translate: SUCCESS → {result}")
                return result
            print(f"  Google Translate: WARNING - empty/None result")
        except Exception as e:
            print(f"  Google Translate: EXCEPTION - {type(e).__name__}: {e}")
        return None


    def set_speech_enabled(self, enabled):
        self.speech_enabled = enabled
        print(f"Speech Enabled: {self.speech_enabled}")
        
    def get_ui_state(self):
        """Thread-safe state snapshot for UI."""
        with self.lock:
            return {
                "sentence": self.last_sentence,
                "english_text": self.last_english_sentence,
                "last_update_id": self.last_update_id,
                "audio": self.last_audio,
                "target_language": self.target_language,
                "speech_enabled": self.speech_enabled,
                "buffer": self.gesture_buffer.get_current_buffer()
            }

    def set_mode(self, mode):
        if mode in ["live", "practice"]:
            self.mode = mode
            print(f"Mode set to: {self.mode}")
            self.gesture_buffer.buffer = []
            self.gesture_buffer.last_gesture = None
            self.last_sentence = ""

    def process_sentence_event(self, text):
        """
        Finalizes a sentence event.
        Translates SYNCHRONOUSLY so translated text appears immediately,
        then spawns background thread only for audio generation.
        """
        # 0. Pre-processing (Grammar + Custom Rules)
        text = self.text_processor.process(text)

        # 1. Translate SYNCHRONOUSLY (blocks ~1-2s but ensures translated text is visible immediately)
        final_english, final_translated = self.translate_text(text)

        # 2. Update state with translated text right away (visible on very next /status poll)
        self.last_english_sentence = final_english
        self.last_sentence = final_translated
        self.last_audio = None    # Clear old audio
        self.last_update_id += 1  # Notify UI of new text event

        print(f"Text ready: [{self.target_language}] {final_translated}")

        # 3. Background Job: AUDIO ONLY (translation already done above)
        def audio_pipeline():
            if not final_translated.strip():
                return
            try:
                tts_lang = self.target_language
                # Use English TTS if translation didn't change the text
                if self.target_language != "English" and final_translated == final_english:
                    tts_lang = "English"

                from src.tts import generate_audio
                audio_data = generate_audio(final_translated, tts_lang)

                with self.lock:
                    self.last_audio = audio_data

            except Exception as e:
                print(f"Background TTS Error: {e}")

            # Notify UI that audio is now available
            with self.lock:
                self.last_update_id += 1
            print(f"Audio ready for: {final_translated}")

        threading.Thread(target=audio_pipeline, daemon=True).start()

        return final_translated

    def simulate_text(self, text):
        """
        Manually trigger the pipeline with a text string.
        """
        print(f"DEBUG: Simulating Text Input: {text}")
        return self.process_sentence_event(text)

    def process_frame(self, frame):
        self.frame_count += 1
        
        # SKIP LOGIC: Return cached result if skipping
        if self.frame_count % (self.skip_frames + 1) != 0:
            return frame, self.last_predicted_text, self.last_confidence

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Face Analysis (DISABLED)
        emotion = "Neutral"

        # 2. Hand Analysis
        results = self.hands.process(rgb_frame)

        # Standard Prediction Logic
        predicted_text = f"Mode: {self.mode.upper()}"
        if self.mode == "live" and self.gesture_buffer.buffer:
           predicted_text += f"\nBuffer: {self.gesture_buffer.get_current_buffer()}"

        confidence = 0.0

        # Logic Loop
        if results.multi_hand_landmarks and results.multi_handedness:
            # hand_count = len(results.multi_hand_landmarks) # Unused
            
            left_hand_data = [0.0] * 42
            right_hand_data = [0.0] * 42
            
            for idx, hand_handedness in enumerate(results.multi_handedness):
                hand_label = hand_handedness.classification[0].label
                landmarks = results.multi_hand_landmarks[idx]
                
                # Visual Feedback: Draw Landmarks
                self.mp_drawing.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                flat_landmarks = []
                for lm in landmarks.landmark:
                    flat_landmarks.append(lm.x)
                    flat_landmarks.append(lm.y)
                
                if hand_label == "Left":
                    left_hand_data = flat_landmarks
                else:
                    right_hand_data = flat_landmarks

            # Merge for Preprocessor (84 features)
            raw_features = left_hand_data + right_hand_data
            
            # --- START NEW TFLITE LOGIC ---
            if self.interpreter:
                # 1. Preprocess & Temporal Buffer
                # returns (1, 10, 84) if ready, else None
                sequence_input = self.preprocessor.process_frame(raw_features, stateful=True)
                
                if sequence_input is not None:
                    # 2. Run Inference
                    try:
                        self.interpreter.set_tensor(self.input_details[0]['index'], sequence_input.astype(np.float32))
                        self.interpreter.invoke()
                        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                        
                        # 3. Prediction & Smoothing
                        class_id = np.argmax(output_data[0])
                        conf = float(output_data[0][class_id])
                        
                        if conf > 0.8: # Require 80%+ accuracy confidence to even consider the frame
                            self.prediction_buffer.append(class_id)
                        else:
                            # If low confidence, maybe append None or keep existing buffer?
                            # For now, just ignore weak frames
                            pass
                            
                        # Majority Vote Smoothing
                        if len(self.prediction_buffer) == self.prediction_buffer.maxlen:
                            counts = Counter(self.prediction_buffer)
                            most_common_id, count = counts.most_common(1)[0]
                            
                            # Require at least 70% consistency across the buffer (14 out of 20 frames)
                            if count >= int(self.prediction_buffer.maxlen * 0.7):
                                current_gesture = GESTURES.get(most_common_id, "Unknown")
                                confidence = conf
                                
                                predicted_text += f"\nDetected: {current_gesture} ({conf:.2f})"
                                
                                if self.mode == "live":
                                    self.gesture_buffer.add_gesture(current_gesture, emotion, hand_count=1)
                                    
                                    # Update display with buffer content
                                    buffered_text = self.gesture_buffer.get_current_buffer()
                                    if buffered_text:
                                        predicted_text += f"\nBuilding: {buffered_text}"
                                        
                    except Exception as e:
                        print(f"Inference Error: {e}")
            # --- END NEW TFLITE LOGIC ---
                                            

            
            # Update hands visible time
            self.last_hands_visible_time = time.time()
            self.sentence_triggered = False

        else:
            # No Hands Detected Logic
            if self.mode == "live" and not self.sentence_triggered:
                elapsed = time.time() - self.last_hands_visible_time
                if elapsed > self.HANDS_OFF_COOLDOWN:
                    # Trigger Speech Finalization
                    final_sentence = self.gesture_buffer.finalize_sentence()
                    if final_sentence:
                        # Use centralized processing logic
                        polished_translated = self.process_sentence_event(final_sentence)
                        
                        predicted_text += f"\nSPEAKING: {polished_translated}"
                    
                    self.sentence_triggered = True
 
            self.prediction_history = []


        # Update Cache
        self.last_predicted_text = predicted_text
        self.last_confidence = confidence
        
        return frame, predicted_text, confidence
