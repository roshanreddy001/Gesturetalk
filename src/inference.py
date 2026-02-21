import cv2
import mediapipe as mp
import numpy as np
import tflite_runtime.interpreter as tflite
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


class DataCollectionState(Enum):
    IDLE = 0
    WARMUP = 1
    RECORDING = 2
    COOLDOWN = 3

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
    10: "Emergency",
    11: "Call Doctor",
    12: "Call Family",
    13: "Hungry",
    14: "Thirsty",
    15: "Washroom",
    16: "Happy",
    17: "Sad",
    18: "Angry",
    19: "Scared",
    20: "Tired",
    21: "Pain",
    22: "Fine",
    23: "Sick",
    24: "Medicine",
    25: "Home",
    26: "Friend",
    27: "Love",
    28: "Time",
    29: "Where",
    30: "Who",
    31: "What",
    32: "Why",
    33: "Money"
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
    "Emergency": "This is an emergency",
    "Call Doctor": "Please call a doctor",
    "Call Family": "Please call my family",
    "Hungry": "I am hungry",
    "Thirsty": "I am thirsty",
    "Washroom": "I need to use the washroom",
    "Happy": "I am happy",
    "Sad": "I am feeling sad",
    "Angry": "I am angry",
    "Scared": "I am scared",
    "Tired": "I am tired",
    "Pain": "I am in pain",
    "Fine": "I am doing fine",
    "Sick": "I am not feeling well",
    "Medicine": "I need medicine",
    "Home": "I want to go home",
    "Friend": "This is my friend",
    "Love": "I love you",
    "Time": "What time is it",
    "Where": "Where is it",
    "Who": "Who is that",
    "What": "What is this",
    "Why": "Why is that",
    "Money": "I need money"
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
        
        # Join with punctuation
        sentence = ". ".join(self.buffer) + "."
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
        self.prediction_buffer = deque(maxlen=5) # Smoothing buffer
        
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
        self.HANDS_OFF_COOLDOWN = 0.5 # Seconds
        self.sentence_triggered = False

        # Optimization: Frame Skipping
        self.frame_count = 0
        self.skip_frames = 0 # Process every frame for smooth video
        self.last_predicted_text = ""
        self.last_confidence = 0.0

        # Data Collection State
        self.collection_mode = False
        self.collection_state = DataCollectionState.IDLE
        self.current_gesture_id = 0
        self.state_start_time = 0
        self.samples_saved = 0
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        self.csv_file = os.path.join(self.data_dir, 'keypoints.csv')
        
        # Ensure data directory exists
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

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


    def get_collection_stats(self):
        gesture_name = GESTURES.get(self.current_gesture_id, "Unknown")
        status_text = self.collection_state.name
        
        # Add localized context to status
        if self.collection_state == DataCollectionState.WARMUP:
            status_text = "GET READY"
        elif self.collection_state == DataCollectionState.RECORDING:
            status_text = "RECORDING"
        elif self.collection_state == DataCollectionState.COOLDOWN:
            status_text = "NEXT GESTURE..."
            
        return {
            "mode": self.collection_mode,
            "gesture": gesture_name,
            "samples": self.samples_saved,
            "status": status_text,
            "gesture_id": self.current_gesture_id,
            "total_gestures": len(GESTURES)
        }

            
    def set_collection_mode(self, enabled):
        self.collection_mode = enabled
        if enabled:
            self.collection_state = DataCollectionState.WARMUP
            self.current_gesture_id = 0
            self.state_start_time = time.time()
            self.samples_saved = 0
            print("Starting Data Collection")
        else:
            self.collection_state = DataCollectionState.IDLE
            print("Stopping Data Collection")

    def save_gesture_data(self, input_vector, label_id):
        # input_vector is a list of 84 floats
        try:
            row = list(input_vector) + [label_id]
            
            # Check if file exists to write header
            file_exists = os.path.exists(self.csv_file)
            
            with open(self.csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    header = []
                    for i in range(21): header.extend([f'lx{i}', f'ly{i}'])
                    for i in range(21): header.extend([f'rx{i}', f'ry{i}'])
                    header.append('label')
                    writer.writerow(header)
                
                writer.writerow(row)
            # print(f"Saved sample for gesture {label_id}") # Uncomment for verbose logging
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            return False

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
                "buffer": self.gesture_buffer.get_current_buffer(),
                "collection_stats": self.get_collection_stats()
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

        # DATA COLLECTION OVERRIDE
        if self.collection_mode:
            current_time = time.time()
            elapsed = current_time - self.state_start_time
            gesture_name = GESTURES.get(self.current_gesture_id, "Unknown")
            
            overlay_text = f"COLLECTION MODE\nGesture: {gesture_name} ({self.current_gesture_id}/{len(GESTURES)-1})"
            
            if self.collection_state == DataCollectionState.WARMUP:
                countdown = 3 - int(elapsed)
                overlay_text += f"\n\nGET READY...\n{countdown}"
                if elapsed >= 3:
                    self.collection_state = DataCollectionState.RECORDING
                    self.state_start_time = current_time
                    self.samples_saved = 0
                    
            elif self.collection_state == DataCollectionState.RECORDING:
                overlay_text += f"\n\nRECORDING...\nSamples: {self.samples_saved}"
                
                # Logic Loop for recording
                if results.multi_hand_landmarks and results.multi_handedness:
                    # Visual Feedback: Draw Landmarks during collection
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    left_hand_data = [0.0] * 42
                    right_hand_data = [0.0] * 42
                    
                    for idx, hand_handedness in enumerate(results.multi_handedness):
                        hand_label = hand_handedness.classification[0].label
                        landmarks = results.multi_hand_landmarks[idx]
                        
                        flat_landmarks = []
                        for lm in landmarks.landmark:
                            flat_landmarks.append(lm.x)
                            flat_landmarks.append(lm.y)
                        
                        if hand_label == "Left":
                            left_hand_data = flat_landmarks
                        else:
                            right_hand_data = flat_landmarks
                            
                    input_vector = left_hand_data + right_hand_data
                    
                    # Save data (throttle if needed, but here we take every processed frame)
                    if any(v != 0 for v in input_vector):
                        self.save_gesture_data(input_vector, self.current_gesture_id)
                        self.samples_saved += 1
                
                # Check for completion (either time or sample count)
                # Using 30 seconds as requested
                if elapsed >= 30:
                    self.collection_state = DataCollectionState.COOLDOWN
                    self.state_start_time = current_time
                    
            elif self.collection_state == DataCollectionState.COOLDOWN:
                overlay_text += f"\n\nNEXT GESTURE IN...\n{2 - int(elapsed)}"
                if elapsed >= 2:
                    self.current_gesture_id += 1
                    if self.current_gesture_id >= len(GESTURES):
                        self.collection_mode = False
                        self.collection_state = DataCollectionState.IDLE
                        overlay_text = "COLLECTION COMPLETE!"
                    else:
                        self.collection_state = DataCollectionState.WARMUP
                        self.state_start_time = current_time

            self.last_predicted_text = overlay_text
            self.last_confidence = 1.0
            return frame, overlay_text, 1.0

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
                        
                        if conf > 0.6: # Lower threshold, rely on smoothing
                            self.prediction_buffer.append(class_id)
                        else:
                            # If low confidence, maybe append None or keep existing buffer?
                            # For now, just ignore weak frames
                            pass
                            
                        # Majority Vote Smoothing
                        if len(self.prediction_buffer) == self.prediction_buffer.maxlen:
                            counts = Counter(self.prediction_buffer)
                            most_common_id, count = counts.most_common(1)[0]
                            
                            # Require 3/5 consistency
                            if count >= 3:
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
