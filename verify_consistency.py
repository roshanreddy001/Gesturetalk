import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from inference import GESTURES as G_INF
    from data_collection import GESTURES as G_COL
    from offline_dict import OFFLINE_TRANSLATIONS
    
    print(f"Inference Gestures: {len(G_INF)}")
    print(f"Collection Gestures: {len(G_COL)}")
    print(f"Offline Dictionary: {len(OFFLINE_TRANSLATIONS)}")
    
    if G_INF != G_COL:
        print("ERROR: Inference and Collection gestures do not match!")
        # Print diff
        for k in G_INF:
            if k not in G_COL: print(f"Missing in COL: {k}")
            elif G_INF[k] != G_COL[k]: print(f"Mismatch: {k} -> {G_INF[k]} vs {G_COL[k]}")
        sys.exit(1)
        
    # Check if all gestures have translations
    missing_trans = []
    for k, v in G_INF.items():
        if v not in OFFLINE_TRANSLATIONS:
            missing_trans.append(v)
            
    if missing_trans:
        print(f"WARNING: The following gestures have no offline translation: {missing_trans}")
    else:
        print("SUCCESS: All gestures have translations.")
        
    print("ALL CHECKS PASSED.")

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
