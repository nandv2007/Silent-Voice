from input_emotion import get_detected_intent
from voice_generator import speak

def main():
    intent_text = get_detected_intent()
    print("Detected Intent:", intent_text)
    speak(intent_text)

if __name__ == "__main__":
    main()

