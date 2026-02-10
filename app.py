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
    # 1. Get thread-safe snapshot from Inference
    ui_state = recognizer.get_ui_state()
    
    # 2. Get local prediction state (thread-safe via app lock)
    with lock:
        pred_text = current_prediction["text"]
        pred_conf = current_prediction["confidence"]
        
        sentence_to_speak = ui_state["sentence"]
        audio_data = None
        
        current_id = ui_state["last_update_id"]
        last_id = getattr(recognizer, 'last_spoken_id', -1)
        
        # Check if this sentence is NEW (ID changed)
        # AND not in collection mode
        is_collecting = ui_state["collection_stats"]["mode"]
        
        if current_id != last_id and sentence_to_speak and not is_collecting:
             if ui_state["audio"]:
                 audio_data = ui_state["audio"]
                 print(f"Sending Pre-generated Audio for: {sentence_to_speak} (ID: {current_id})")
             else:
                 print(f"Audio not ready yet for: {sentence_to_speak} (ID: {current_id})")
            
             recognizer.last_spoken_id = current_id # Mark this ID as processed
            
    response = {
        "text": pred_text,
        "confidence": float(pred_conf),
        "sentence": sentence_to_speak,
        
        # New Fields from Snapshot
        "buffer": ui_state["buffer"],
        "english_text": str(ui_state["english_text"]),
        "translated_text": str(ui_state["sentence"]),
        "target_language": ui_state["target_language"],
        "language": ui_state["target_language"],
        "audio": audio_data,
        
        # State Flags for UI Sync
        "speech_enabled": ui_state["speech_enabled"],

        # Data Collection Stats
        "collection_stats": ui_state["collection_stats"]
    }
    return jsonify(response)

@app.route('/set_language', methods=['POST'])
def set_language():
    data = request.json
    language = data.get('language', 'English')
    recognizer.set_language(language)
    return jsonify({"status": "success", "language": language})

@app.route('/toggle_speech', methods=['POST'])
def toggle_speech():
    data = request.json
    enabled = data.get('enabled', True)
    recognizer.set_speech_enabled(enabled)
    return jsonify({"status": "success", "enabled": enabled})

@app.route('/toggle_collection', methods=['POST'])
def toggle_collection():
    data = request.json
    enabled = data.get('enabled', False)
    recognizer.set_collection_mode(enabled)
    return jsonify({"status": "success", "mode": "collection" if enabled else "live"})

@app.route('/simulate_input', methods=['POST'])
def simulate_input():
    data = request.json
    text = data.get('text', '')
    if text:
        recognizer.simulate_text(text)
    return jsonify({"status": "success", "text": text})

@app.route('/test_audio', methods=['POST'])
def test_audio():
    # Generate a simple test sound (e.g., "Hello, system check.")
    try:
        from src.tts import generate_audio
        audio_data = generate_audio("System check. Audio operational.", "English")
        return jsonify({"status": "success", "audio": audio_data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
