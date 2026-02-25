import itertools
import random
import csv
import os

# The 15 core gestures
GESTURES = [
    "Hello", "Goodbye", "Yes", "No", "Please", "ThankYou", "Sorry",
    "Help", "Stop", "Wait", "Hungry", "Water", "Pain", "Emergency", "Home"
]

# Output path
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

CSV_FILE = os.path.join(DATA_DIR, 'gesture_sentences.csv')

# Predefined templates to map gesture sequences to natural English
# Using wildcards where combinations make sense
TEMPLATES = {
    # 2 GESTURE COMBINATIONS
    ("Hello", "Please"): "Hello, can you please assist me?",
    ("Hello", "Help"): "Hello, please help me.",
    ("Hello", "Yes"): "Hello, yes I agree.",
    ("Hello", "No"): "Hello, no thank you.",
    ("Goodbye", "ThankYou"): "Goodbye, and thank you.",
    ("Yes", "Please"): "Yes, please.",
    ("No", "ThankYou"): "No, thank you.",
    ("Please", "Wait"): "Please wait a moment.",
    ("Please", "Stop"): "Please stop what you are doing.",
    ("Please", "Help"): "Can you please help me?",
    ("Sorry", "Yes"): "I am sorry, but yes.",
    ("Sorry", "No"): "I am sorry, but no.",
    ("Sorry", "Wait"): "Sorry, please wait.",
    ("Hungry", "Please"): "I am hungry, please get me food.",
    ("Water", "Please"): "I am thirsty, please give me water.",
    ("Pain", "Help"): "I am in pain, please help me.",
    ("Emergency", "Help"): "This is an emergency, I need help immediately!",
    ("Emergency", "Stop"): "Stop, this is an emergency!",
    ("Home", "Please"): "Please take me home.",
    
    # 3 GESTURE COMBINATIONS
    ("Hello", "Please", "Help"): "Hello, can you please help me?",
    ("Hello", "Hungry", "Please"): "Hello, I am hungry, please.",
    ("Hello", "Water", "Please"): "Hello, I need water please.",
    ("Sorry", "Please", "Wait"): "Sorry, can you please wait?",
    ("Please", "Stop", "Pain"): "Please stop, it causes me pain.",
    ("Yes", "Home", "Please"): "Yes, I want to go home please.",
    ("No", "Stop", "Please"): "No, please stop.",
    ("Emergency", "Pain", "Help"): "It's an emergency, I am in pain, please help me!",
}


def generate_dataset(target_count=3500):
    dataset = {}
    
    # 1. Add all direct templates
    for combo, sentence in TEMPLATES.items():
        key = "+".join(combo)
        dataset[key] = sentence

    # 2. Add individual base gestures
    BASE_PHRASES = {
        "Hello": "Hello there.",
        "Goodbye": "Goodbye, see you later.",
        "Yes": "Yes, I agree.",
        "No": "No, I disagree.",
        "Please": "Please.",
        "ThankYou": "Thank you very much.",
        "Sorry": "I am sorry.",
        "Help": "Please help me.",
        "Stop": "Stop that immediately.",
        "Wait": "Please wait here.",
        "Hungry": "I am feeling hungry.",
        "Water": "I would like some water.",
        "Pain": "I am currently in pain.",
        "Emergency": "This is an emergency!",
        "Home": "I want to go back home."
    }
    for g, s in BASE_PHRASES.items():
        dataset[g] = s

    # 3. Procedural Generation to reach ~3500
    # Rule based mapping for random combinations
    print(f"Base size: {len(dataset)}. Proceeding to generation...")
    
    # Generate length 2, 3, and 4 combos
    all_combinations = []
    
    # length 2: 15 * 14 = 210
    all_combinations.extend(list(itertools.permutations(GESTURES, 2)))
    # length 3: 15 * 14 * 13 = 2730
    all_combinations.extend(list(itertools.permutations(GESTURES, 3)))
    # length 4: sampled heavily to fill remainder
    all_4_combos = list(itertools.permutations(GESTURES, 4))
    random.shuffle(all_4_combos)
    
    # Add length 4 until we have enough
    all_combinations.extend(all_4_combos[:target_count]) 

    # Some helper mappings to construct sentences programmatically
    phrase_bits = {
        "Hello": ["Hello,", "Hi,", "Greetings,", "Hey,"],
        "Goodbye": ["Goodbye.", "Bye.", "See you.", "Leaving now."],
        "Yes": ["Yes,", "Yeah,", "Indeed,", "Correct,"],
        "No": ["No,", "Nope,", "Negative,", "Incorrect,"],
        "Please": ["please", "kindly", "if you would"],
        "ThankYou": ["thank you.", "thanks.", "much appreciated."],
        "Sorry": ["Sorry.", "I apologize.", "My apologies."],
        "Help": ["help me.", "I need assistance.", "assist me."],
        "Stop": ["stop.", "halt.", "wait right there."],
        "Wait": ["wait a moment.", "hold on.", "give me a second."],
        "Hungry": ["I'm hungry.", "I need food.", "get me something to eat."],
        "Water": ["I need water.", "I'm thirsty.", "get me a drink."],
        "Pain": ["I'm in pain.", "it hurts.", "I am suffering."],
        "Emergency": ["This is an emergency!", "Urgent!", "Emergency situation!"],
        "Home": ["I want to go home.", "take me home.", "let's go home."]
    }

    count = len(dataset)
    for combo in all_combinations:
        key = "+".join(combo)
        if key in dataset:
            continue
            
        if count >= target_count:
            break
            
        # Construct a semi-natural sentence based on the sequence 
        # (simulating a basic sequence-to-sequence thought process)
        sentence_parts = []
        for word in combo:
            choices = phrase_bits[word]
            sentence_parts.append(random.choice(choices))
            
        # Join pieces
        raw_sentence = " ".join(sentence_parts)
        
        # Clean up punctuation (e.g. multiple capitals/periods due to basic joining)
        # This simulates that sequences can sometimes be choppy if completely random,
        # but the LanguageTool integration in inference will smooth it further.
        clean_sentence = raw_sentence.replace(",.", ".").replace("..", ".").replace("!.", "!")
        clean_sentence = clean_sentence[0].upper() + clean_sentence[1:]
        
        dataset[key] = clean_sentence
        count += 1
        
    print(f"Final dataset size: {len(dataset)}")

    # Write to CSV
    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Gesture_Sequence', 'Natural_Sentence'])
        for k, v in dataset.items():
            writer.writerow([k, v])
            
    print(f"Successfully generated {len(dataset)} examples in {CSV_FILE}")

if __name__ == "__main__":
    generate_dataset()
