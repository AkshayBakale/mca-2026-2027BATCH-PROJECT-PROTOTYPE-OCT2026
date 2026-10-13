import torch          # keep torch first (avoids DLL-order issues on Windows)
import os
import time
import cv2
import mediapipe as mp
import joblib
import numpy as np
import pyttsx3
from collections import deque

from nn_models import AutoEncoder, MotionNet
from features import normalize
from word_suggester import suggest_words
from dictionary_lookup import get_definition
from recommender import recommend_similar_signs

UNMIRROR = False    # set to True if your letters come out wrong

# ---------- Text-to-speech ----------
tts = pyttsx3.init()
tts.setProperty("rate", 150)


def speak(text):
    if text:
        tts.say(text.lower())
        tts.runAndWait()          # video pauses briefly while speaking


# ---------- Load models ----------
clf = joblib.load("models/asl_model.pkl")
scaler = joblib.load("models/autoencoder_scaler.pkl")
threshold = joblib.load("models/autoencoder_threshold.pkl")

ae = AutoEncoder()
ae.load_state_dict(torch.load("models/autoencoder.pt"))
ae.eval()

motion_model, motion_classes, SEQ_LEN = None, [], 30
if os.path.exists("models/motion_model.pt"):
    ckpt = torch.load("models/motion_model.pt")
    motion_classes, SEQ_LEN = ckpt["classes"], ckpt["seq_len"]
    motion_model = MotionNet(len(motion_classes), ckpt["cell"], ckpt["bidirectional"])
    motion_model.load_state_dict(ckpt["state_dict"])
    motion_model.eval()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# ---------- State ----------
caption, current_word, definition_text = "", "", ""
suggestions, similar = [], []
last_added, last_time = "", time.time()
STABLE_TIME = 1.2
mode = "STATIC"
seq_buffer = deque(maxlen=SEQ_LEN)
last_motion_time = 0.0
motion_text = ""


def prev_word():
    parts = caption.split()
    return parts[-1] if parts else None


def commit(token):
    global caption, current_word, suggestions, definition_text
    if token == "SPACE":
        if current_word:
            caption += current_word + " "
            definition_text = get_definition(current_word)
            current_word = ""
        suggestions = []
    elif token == "DEL":
        if current_word:
            current_word = current_word[:-1]
        else:
            caption = caption[:-1]
        suggestions = suggest_words(current_word, prev_word()) if current_word else []
    elif token in ("NOTHING", "UNKNOWN"):
        pass
    elif len(token) == 1:                                  # a letter
        current_word += token
        suggestions = suggest_words(current_word, prev_word())
    else:                                                  # a whole-word motion sign (e.g. HELLO)
        if current_word:
            caption += current_word + " "
            current_word = ""
        caption += token + " "
        definition_text = get_definition(token)
        suggestions = []


print("Controls: 'm' = STATIC/MOTION mode | 's' = speak caption | 'c' = clear | 'q' = quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    row = None
    if result.multi_hand_landmarks:
        hl = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
        row = np.array([[lm.x, lm.y, lm.z] for lm in hl.landmark], dtype=np.float32).flatten()

    if mode == "STATIC":
        if row is not None:
            row2 = normalize(row.reshape(1, -1))
            if UNMIRROR:
                row2[:, 0::3] *= -1                        # undo the webcam mirror effect
            xs = torch.tensor(scaler.transform(row2).astype(np.float32))
            with torch.no_grad():
                error = torch.mean((ae(xs) - xs) ** 2).item()      # autoencoder anomaly check

            current_pred = "UNKNOWN" if error > threshold else str(clf.predict(row2)[0]).upper()
            similar = recommend_similar_signs(current_pred) if len(current_pred) == 1 else []

            color = (0, 0, 255) if current_pred == "UNKNOWN" else (0, 255, 0)
            cv2.putText(frame, f"Detected: {current_pred}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            if similar:
                cv2.putText(frame, "Similar signs: " + ", ".join(similar), (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)

            if current_pred != last_added and time.time() - last_time > STABLE_TIME:
                commit(current_pred)
                last_added, last_time = current_pred, time.time()
        else:
            last_added, similar = "", []

    else:  # MOTION mode: LSTM/GRU over the last SEQ_LEN frames (raw landmarks, as recorded)
        seq_buffer.append(row if row is not None else np.zeros(63, dtype=np.float32))
        if motion_model is None:
            motion_text = "No motion model - run train_lstm.py"
        elif len(seq_buffer) == SEQ_LEN and row is not None and time.time() - last_motion_time > 2.0:
            x = torch.tensor(np.array(seq_buffer)[None, ...], dtype=torch.float32)
            with torch.no_grad():
                probs = torch.softmax(motion_model(x), dim=1)[0]
            conf, idx = probs.max(0)
            label = motion_classes[idx.item()]
            motion_text = f"Motion: {label} ({conf.item():.2f})"
            if conf.item() > 0.9 and label != "NOTHING":
                if label == "SPEAK":
                    speak((caption + current_word).strip())
                else:
                    commit(label)
                seq_buffer.clear()
                last_motion_time = time.time()
        cv2.putText(frame, motion_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame, f"Mode: {mode} (press m)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # ---------- Caption bar ----------
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 90), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, (caption + current_word)[-60:], (10, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if suggestions:
        cv2.putText(frame, "Suggestions: " + ", ".join(suggestions), (10, h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    if definition_text:
        cv2.putText(frame, definition_text[:70], (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("ASL Recognition - Live Captioning", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        caption, current_word, definition_text, suggestions = "", "", "", []
    elif key == ord('s'):
        speak((caption + current_word).strip())
    elif key == ord('m'):
        mode = "MOTION" if mode == "STATIC" else "STATIC"
        seq_buffer.clear()
        last_added = ""

cap.release()
cv2.destroyAllWindows()