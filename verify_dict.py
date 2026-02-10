import sys
import os

# Robustly find src based on script location
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, 'src')
sys.path.append(src_path)
print(f"Added {src_path} to sys.path")

try:
    from offline_dict import OFFLINE_TRANSLATIONS
    print(f"Successfully imported OFFLINE_TRANSLATIONS. Total keys: {len(OFFLINE_TRANSLATIONS)}")
    print("Keys found:")
    for key in OFFLINE_TRANSLATIONS:
        print(f"- {key}")
    
    # Check for duplicates (Python dicts don't allow duplicate keys, but we can check if we accidentally overwrote anything)
    expected_keys = [
        "Hello", "Goodbye", "Yes", "No", "Please", "Thank You", "Sorry",
        "Help", "Stop", "Wait", "Emergency", "Call Doctor", "Call Family",
        "Hungry", "Thirsty", "Washroom", "Happy", "Sad", "Angry", "Scared",
        "Tired", "Pain", "Fine", "Sick"
    ]
    
    found_keys = list(OFFLINE_TRANSLATIONS.keys())
    missing = [k for k in expected_keys if k not in found_keys]
    
    if missing:
        print(f"\nMISSING KEYS: {missing}")
        sys.exit(1)
    else:
        print("\nAll expected keys are present.")

except Exception as e:
    print(f"Error importing or verifying dictionary: {e}")
    sys.exit(1)
