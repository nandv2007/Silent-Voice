import cv2
import mediapipe as mp
import pyttsx3
import numpy as np
from collections import deque, Counter
import time
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

engine = pyttsx3.init()
engine.setProperty('rate', 150)

VOTE_LEN = 7
VOTE_MIN_COUNT = 4
THUMB_EXTENDED_DIST = 0.5
FINGER_EXTENDED_MARGIN = 0
SPEAK_COOLDOWN = 1.2


hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6)
cap = cv2.VideoCapture(0)

pred_votes = deque(maxlen=VOTE_LEN)
last_spoken = ""
last_spoken_time = 0.0

def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def finger_states_from_landmarks(landmarks):
    
    tips = [4,8,12,16,20]
    wrist = landmarks[0]
    states = []

    thumb_tip = landmarks[tips[0]]
    
    td = dist(thumb_tip, wrist)
    thumb_extended = 1 if td > THUMB_EXTENDED_DIST else 0
    states.append(thumb_extended)

    for t in tips[1:]:
        tip = landmarks[t]
        pip = landmarks[t-2]
        states.append(1 if tip.y < pip.y else 0)

    return states 

def detect_gesture_from_states(states, landmarks):
    
    total = sum(states)
    thumb, idx, mid, ring, pinky = states

    if total == 5:
        return "Hello"
    if total == 0:
        return "Okay"
    if total == 1 and idx == 1:
        return "Yes"
    if total == 2 and idx == 1 and mid == 1:
        return "No"
    if total == 4 and thumb == 0 and idx==1 and mid==1 and ring==1 and pinky==1:
        
        thumb_tip = landmarks[4]
        index_mcp = landmarks[5]
        if dist(thumb_tip, index_mcp) < 0.1:
            return "Thank you"  
    
    if thumb == 1 and idx == 0 and mid == 0 and ring == 0 and pinky == 1:
        return "I like this"
    
    if thumb == 1 and idx + mid + ring + pinky == 0:
        return "Good"
    
    if idx == 1 and mid == 1 and ring == 1 and thumb == 0 and pinky == 0:
        return "Please wait"
    
    if mid == 1 and ring == 1 and pinky == 1 and thumb == 0 and idx == 0:
        return "Help me"
    
    if idx == 1 and pinky == 1 and mid == 0 and ring == 0:
        return "Need water"
    
    return ""

def speak_if_needed(pred):
    global last_spoken, last_spoken_time
    if not pred:
        return
    t = time.time()
    
    if pred != last_spoken or (t - last_spoken_time) > SPEAK_COOLDOWN:
        engine.say(pred)
        engine.runAndWait()
        last_spoken = pred
        last_spoken_time = t

def draw_overlays(frame, hand_label, states, current_pred, votes_counter):
    h, w, _ = frame.shape
    
    cv2.rectangle(frame, (0,0), (w, 80), (0,0,0), -1)
    cv2.putText(frame, "SilentVoice-Hand gesture recognition", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"Hand: {hand_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
    cv2.putText(frame, f"States T I M R P: {states}", (220, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

    color = (0,200,0) if current_pred else (0,120,255)
    cv2.putText(frame, f"Pred: {current_pred}", (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    y0 = 100
    for i, (lab, cnt) in enumerate(votes_counter.most_common(5)):
        cv2.putText(frame, f"{lab}: {cnt}", (10, y0 + i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 2)

if not cap.isOpened():
    print("Error: cannot open webcam")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    hand_label = "None"
    states = [0,0,0,0,0]
    gesture = ""

    if results.multi_hand_landmarks:
        if results.multi_handedness:
            hand_label = results.multi_handedness[0].classification[0].label  
        handLms = results.multi_hand_landmarks[0]
        
        lm = handLms.landmark
        states = finger_states_from_landmarks(lm)
        gesture = detect_gesture_from_states(states, lm)
        mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    pred_for_vote = gesture if gesture else "NONE"
    pred_votes.append(pred_for_vote)
    votes_counter = Counter(pred_votes)

    top_label, top_count = votes_counter.most_common(1)[0]

    current_prediction = top_label if top_label != "NONE" and top_count >= VOTE_MIN_COUNT else ""


    if current_prediction:
        speak_if_needed(current_prediction)


    display = cv2.flip(frame, 1)

    draw_overlays(display, hand_label, states, current_prediction, votes_counter)

    cv2.imshow("SilentVoice.py", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if key == ord('c'):

        print("DEBUG STATES:", states, "VOTES:", list(pred_votes))
    if key == ord('+'):

        THUMB_EXTENDED_DIST += 0.005
        print("THUMB_EXTENDED_DIST ->", THUMB_EXTENDED_DIST)
    if key == ord('-'):
        THUMB_EXTENDED_DIST = max(0.01, THUMB_EXTENDED_DIST - 0.005)
        print("THUMB_EXTENDED_DIST ->", THUMB_EXTENDED_DIST)

cap.release()
cv2.destroyAllWindows()
