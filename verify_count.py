import sys
import os

# Robustly find src based on script location
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, 'src')
sys.path.append(src_path)

try:
    from offline_dict import OFFLINE_TRANSLATIONS
    print(f"Total keys: {len(OFFLINE_TRANSLATIONS)}")
    print("Keys:")
    for k in OFFLINE_TRANSLATIONS:
        print(f" - {k}")

    # Expected count logic: 24 original + 10 new = 34
    if len(OFFLINE_TRANSLATIONS) >= 34:
        print("\nSUCCESS: Dictionary expanded successfully.")
    else:
        print(f"\nWARNING: Count looks low. Expected ~34, got {len(OFFLINE_TRANSLATIONS)}")

except Exception as e:
    print(f"Error: {e}")
