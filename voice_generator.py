import pyttsx3

def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150) 
    engine.setProperty('volume', 1) 
    engine.say(text)
    engine.runAndWait()

