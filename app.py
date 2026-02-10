from flask import Flask, render_template, Response, jsonify, request
import cv2
from src.inference import GestureRecognizer
import threading

app = Flask(__name__)
recognizer = GestureRecognizer()
camera = cv2.VideoCapture(0)

# Global variable to store current prediction to share between threads
current_prediction = {
    "text": "Waiting...",
    "confidence": 0.0
}
lock = threading.Lock()

def gen_frames():
    global current_prediction
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Mirror the frame
        frame = cv2.flip(frame, 1)
        
        # Process frame
        try:
            frame, text, conf = recognizer.process_frame(frame)
        except Exception as e:
            print(f"CRITICAL ERROR in process_frame: {e}")
            frame, text, conf = frame, "Error: " + str(e), 0.0
        
        # Update global state
        with lock:
            current_prediction["text"] = text
            current_prediction["confidence"] = conf

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    with lock:
        # Get pending sentence
        sentence_to_speak = recognizer.last_sentence
        audio_data = None
        
        # Logic FIX: Use Update ID to track new events
        current_id = getattr(recognizer, 'last_update_id', 0)
        last_id = getattr(recognizer, 'last_spoken_id', -1)
        
        last_id = getattr(recognizer, 'last_spoken_id', -1)
        
            # Check if this sentence is NEW (ID changed)
        # CRITICAL FIX: Do NOT generate speech if in Collection Mode
        if current_id != last_id and sentence_to_speak and not recognizer.collection_mode:
             # Audio is now generated in background thread in inference.py
             # We just retrieve it here if available
             if recognizer.last_audio:
                 audio_data = recognizer.last_audio
                 print(f"Sending Pre-generated Audio for: {sentence_to_speak} (ID: {current_id})")
             else:
                 print(f"Audio not ready yet for: {sentence_to_speak} (ID: {current_id})")
            
             recognizer.last_spoken_id = current_id # Mark this ID as processed
            
        response = {
            "text": current_prediction["text"],
            "confidence": float(current_prediction["confidence"]),
            "sentence": sentence_to_speak,
            
             # New Fields for UI
            "buffer": recognizer.gesture_buffer.get_current_buffer(),
            "english_text": recognizer.last_english_sentence if recognizer.last_english_sentence else "...",
            "translated_text": recognizer.last_sentence if recognizer.last_sentence else "...",
            "target_language": recognizer.target_language,
            "language": recognizer.target_language,
            "audio": audio_data,
            
            # State Flags for UI Sync
            "speech_enabled": recognizer.speech_enabled,

             # Data Collection Stats
            "collection_stats": recognizer.get_collection_stats()
        }
        return jsonify(response)

@app.route('/toggle_speech', methods=['POST'])
def toggle_speech():
    data = request.json
    enabled = data.get('enabled', True)
    recognizer.set_speech_enabled(enabled)
    return jsonify({"status": "success", "enabled": enabled})

@app.route('/set_mode', methods=['POST'])
def set_mode():
    data = request.json
    mode = data.get('mode', 'live')
    recognizer.set_mode(mode)
    return jsonify({"status": "success", "mode": mode})

@app.route('/toggle_collection', methods=['POST'])
def toggle_collection():
    data = request.json
    enabled = data.get('enabled', False)
    recognizer.set_collection_mode(enabled)
    return jsonify({"status": "success", "enabled": enabled})


@app.route('/set_language', methods=['POST'])
def set_language():
    data = request.json
    language = data.get('language', 'English')
    recognizer.set_language(language)
    return jsonify({"status": "success", "language": language})

@app.route('/simulate_input', methods=['POST'])
def simulate_input():
    data = request.json
    text = data.get('text', '')
    if text:
        # Move outside lock to avoid blocking /status during API calls
        result = recognizer.simulate_text(text)
        return jsonify({"status": "success", "processed": result})
    return jsonify({"status": "error", "message": "Empty text"})

@app.route('/test_audio', methods=['POST'])
def test_audio():
    # Force English test sound
    text = "System Audio Check. Speakers are working."
    audio_data = generate_audio(text, "English")
    
    # Manually trigger audio update via status loop (optional) or just return success
    # For now, let's just speak it via the status loop hack or return it directly?
    # The frontend expects a void return and relies on separate play mechanism? 
    # WAIT: Frontend's testAudio function calls this but doesn't handle response to play audio. 
    # Frontend logic: fetch('/test_audio', { method: 'POST' }); 
    # It expects nothing. But maybe we should inject it into the status loop?
    
    
    # Store in last_sentence too (so status loop sees it? No, avoid double speech)
    # recognizer.last_sentence = text # Commented out to prevent double-speak
    
    return jsonify({"status": "success", "audio": audio_data})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False) 
