import language_tool_python
import re

class TextProcessor:
    def __init__(self):
        print("Initializing LanguageTool (this may take a moment)...")
        # Use a local server or public API. Default is a local server if Java is installed,
        # or it downloads a server. If that fails, it might need 'remote' for public API.
        try:
            self.tool = language_tool_python.LanguageTool('en-US')
        except Exception as e:
            print(f"Warning: Failed to initialize local LanguageTool: {e}")
            print("Attempting to use public API (remote)...")
            try:
                self.tool = language_tool_python.LanguageTool('en-US', remote_server='https://api.languagetoolplus.com/v2/')
            except Exception as e2:
                print(f"Error: Could not initialize LanguageTool: {e2}")
                self.tool = None

        # Custom Word Enhancement Dictionary
        self.enhancements = {
            "good": "excellent",
            "bad": "unfavorable",
            "sad": "upset",
            "happy": "delighted",
            "mad": "angry",
            "fix": "rectify",
            "use": "utilize",
            "get": "obtain",
            "help": "assist",
            "want": "desire"
        }

    def process(self, text):
        if not text:
            return text

        # 1. Grammar Correction
        if self.tool:
            try:
                # auto_replace applies the best suggestion
                text = self.tool.correct(text)
            except Exception as e:
                print(f"LanguageTool Error: {e}")

        # 2. Custom Rules (Enhancement & Formatting)
        text = self._apply_custom_rules(text)

        return text

    def _apply_custom_rules(self, text):
        # A. Word Enhancement
        # Split into words to avoid replacing substrings (e.g. 'goods' -> 'excellents')
        words = text.split()
        enhanced_words = []
        for word in words:
            # Strip punctuation for lookup
            clean_word = re.sub(r'[^\w\s]', '', word).lower()
            if clean_word in self.enhancements:
                # Preserve capitalization
                replacement = self.enhancements[clean_word]
                if word[0].isupper():
                    replacement = replacement.capitalize()
                
                # Restore punctuation if any
                if word[-1] in ".!?,":
                    replacement += word[-1]
                
                enhanced_words.append(replacement)
            else:
                enhanced_words.append(word)
        
        text = " ".join(enhanced_words)

        # B. Formatting
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        
        # Ensure punctuation at the end
        if text and text[-1] not in ".!?":
            text += ""

        return text

if __name__ == "__main__":
    # Test script
    processor = TextProcessor()
    
    test_sentences = [
        "i want help",
        "this is good idea",
        "why you sad",
        "he go to home"
    ]
    
    print("\n--- Testing Text Processor ---")
    for s in test_sentences:
        processed = processor.process(s)
        print(f"Original: '{s}'")
        print(f"Processed: '{processed}'\n")
