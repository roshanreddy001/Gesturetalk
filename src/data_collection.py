import cv2
import mediapipe as mp
import csv
import os
import numpy as np
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) # Enable 2 Hands

# Directory to save data
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

CSV_FILE = os.path.join(DATA_DIR, 'keypoints.csv')

# --- DEFINED GESTURES (15 CLASSES) ---
# Each gesture matches the ID expected by the sentence generator.
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

# Add reverse mapping for easy lookup by name
NAME_TO_ID = {v: k for k, v in GESTURES.items()}

def init_csv():
    """
    Initializes the CSV file with 84 feature columns:
    - 0-41: Left Hand (x,y for 21 landmarks)
    - 42-83: Right Hand (x,y for 21 landmarks)
    - label: Gesture ID
    """
    if os.path.exists(CSV_FILE):
        # Backup old file if it exists (simple rename to avoid overwrite specific valid data)
        # But for this task, we assume user wants fresh start or we append if format matches.
        # Since format changes, we should ideally start fresh.
        try:
             # Check header to see if it matches
             with open(CSV_FILE, 'r') as f:
                 header = f.readline().strip().split(',')
                 if len(header) != 85: # 84 features + 1 label
                     print("Old CSV format detected. Renaming to keypoints_old.csv")
                     f.close()
                     os.rename(CSV_FILE, os.path.join(DATA_DIR, 'keypoints_old.csv'))
        except Exception as e:
            print(f"Warning checking CSV: {e}")

    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            header = []
            # Left Hand (0-20)
            for i in range(21):
                header.append(f'lx{i}')
                header.append(f'ly{i}')
            # Right Hand (0-20)
            for i in range(21):
                header.append(f'rx{i}')
                header.append(f'ry{i}')
            header.append('label')
            writer.writerow(header)

def save_instance(results, label):
    """
    Extracts landmarks from both hands, standardizes Left vs Right, and saves row.
    """
    # Initialize zero vectors for both hands
    left_hand_data = [0.0] * 42
    right_hand_data = [0.0] * 42

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_handedness in enumerate(results.multi_handedness):
            # MediaPipe's 'Left' label often corresponds to the viewer's 'Right' in selfie mode
            # But the label usually means "This is a Left Hand".
            # Let's trust the label classification.
            
            hand_label = hand_handedness.classification[0].label # "Left" or "Right"
            landmarks = results.multi_hand_landmarks[idx]
            
            flat_landmarks = []
            for lm in landmarks.landmark:
                flat_landmarks.append(lm.x)
                flat_landmarks.append(lm.y)
            
            if hand_label == "Left":
                left_hand_data = flat_landmarks
            else:
                right_hand_data = flat_landmarks

    # Combine: [Left Hand Data] + [Right Hand Data]
    row = left_hand_data + right_hand_data
    row.append(label)
    
    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
    print(f"Saved: {GESTURES[label]}")

def main():
    init_csv()
    cap = cv2.VideoCapture(0)
    
    current_label_id = 0 # Start recording "Hello"
    rec_mode = False
    
    # State tracking for automated collection
    SAMPLES_NEEDED = 300
    current_samples = 0
    in_cooldown = False
    cooldown_start = 0
    COOLDOWN_TIME = 3.0  # seconds between gestures (gives time to switch)
    
    print("--- DUAL HAND DATA COLLECTION ---")
    print("Automated Collection Mode (900 samples per gesture)")
    print("Press 'r' to Start/Pause Automated Recording.")
    print("Press 'q' to Quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame for mirror view
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = hands.process(rgb_frame)
        
        # Draw Stuff
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Label the hand on screen
                label_text = handedness.classification[0].label
                coords = tuple(np.multiply(
                    np.array((hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y)), 
                    [640, 480]).astype(int))
                cv2.putText(frame, label_text, coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # UI Instructions
        if current_label_id < len(GESTURES):
            cv2.putText(frame, f"Target: {GESTURES[current_label_id]} (ID: {current_label_id})", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {current_samples}/{SAMPLES_NEEDED}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Show manual next instructions
            if current_samples >= SAMPLES_NEEDED and not rec_mode:
                cv2.putText(frame, "DONE! Press 'n' to go to NEXT.", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        else:
            cv2.putText(frame, "ALL GESTURES COMPLETE!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            rec_mode = False

        cv2.imshow('Data Collection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Control Logic
        if key == ord('q'):
            break
        elif key == ord('n'):
            current_label_id = (current_label_id + 1) % len(GESTURES)
            current_samples = 0
            rec_mode = False # Ensure recording doesn't auto-start
        elif key == ord('p'):
            current_label_id = (current_label_id - 1) % len(GESTURES)
            current_samples = 0
            rec_mode = False
        elif key == ord('r'): # Toggle Recording
            if current_label_id < len(GESTURES) and current_samples < SAMPLES_NEEDED:
                rec_mode = not rec_mode
                if rec_mode:
                    print(f"Started recording for {GESTURES[current_label_id]}")
                    in_cooldown = True
                    cooldown_start = time.time()
                else:
                    print(f"Paused recording.")

        # Automated Recording Logic
        if rec_mode and current_label_id < len(GESTURES):
            if in_cooldown:
                elapsed = time.time() - cooldown_start
                remaining = max(0, COOLDOWN_TIME - elapsed)
                cv2.putText(frame, f"GET READY: {remaining:.1f}s", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                if remaining <= 0:
                    in_cooldown = False
            else:
                cv2.putText(frame, "RECORDING...", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                # Save every frame directly to hit 300 samples quickly (about 10 seconds at 30 fps)
                if results.multi_hand_landmarks and results.multi_handedness:
                    save_instance(results, current_label_id)
                    current_samples += 1

                # Check if gesture is complete
                if current_samples >= SAMPLES_NEEDED:
                    print(f"Completed {SAMPLES_NEEDED} samples for {GESTURES[current_label_id]}")
                    rec_mode = False # Stop recording to let CPU cool down
                    if current_label_id >= len(GESTURES) - 1:
                        print("Collection entirely finished!")
                    else:
                        print("Auto-paused. Take a break. Press 'n' for next gesture, then 'r' to record.")
        
        # Update display again to show late texts
        cv2.imshow('Data Collection', frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
