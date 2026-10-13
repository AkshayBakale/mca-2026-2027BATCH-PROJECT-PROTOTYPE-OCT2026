import cv2
import mediapipe as mp
import numpy as np
import os

SEQ_LEN = 30                      # ~1 second of video
SAVE_DIR = "data/sequences"
os.makedirs(SAVE_DIR, exist_ok=True)

LABEL = input("Label for this motion sign (no underscores, e.g. HELLO, THANKYOU, J, Z): ").strip().upper()
count = len([f for f in os.listdir(SAVE_DIR) if f.startswith(LABEL + "_")])

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

recording, buffer = False, []
print("Press 'r' to record one sequence (~1 sec), 'q' to quit. Aim for 30+ samples per sign.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    row = np.zeros(63, dtype=np.float32)
    if result.multi_hand_landmarks:
        hl = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
        row = np.array([[lm.x, lm.y, lm.z] for lm in hl.landmark], dtype=np.float32).flatten()

    if recording:
        buffer.append(row)
        if len(buffer) == SEQ_LEN:
            np.save(os.path.join(SAVE_DIR, f"{LABEL}_{count}.npy"), np.array(buffer))
            count += 1
            recording, buffer = False, []
            print(f"Saved sample #{count} for {LABEL}")

    status = "RECORDING..." if recording else "Press 'r' to record"
    cv2.putText(frame, f"{LABEL}: {count} samples | {status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if recording else (0, 255, 0), 2)
    cv2.imshow("Collect motion sequences", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        recording, buffer = True, []
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()