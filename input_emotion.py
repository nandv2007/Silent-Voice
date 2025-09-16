def get_detected_intent():    
    print("Simulating AI: Choose your intent")
    print("1. I need water")
    print("2. I am feeling pain")
    print("3. I want help")
    print("4. I am hungry")
    choice = input("Enter choice (1-4): ")
    mapping = {
        "1": "I need water",
        "2": "I am feeling pain",
        "3": "I want help",
        "4": "I am hungry"
    }
    return mapping.get(choice, "Sorry, couldn't detect intent")

