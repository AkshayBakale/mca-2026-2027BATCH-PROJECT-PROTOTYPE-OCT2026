
import os
import sys
import time
import threading
from collections import deque

import cv2
import joblib
import mediapipe as mp
import numpy as np
import torch
from flask import Flask, Response, jsonify, request, render_template_string

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")

os.chdir(ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from features import normalize
from nn_models import AutoEncoder, MotionNet

try:
    from word_suggester import suggest_words
except Exception:
    suggest_words = None

try:
    from recommender import recommend_similar_signs
except Exception:
    recommend_similar_signs = None

app = Flask(__name__)

# Find files whether your folder is named "model" or "models".
MODEL_DIR = None
for folder in ("models", "model"):
    candidate = os.path.join(ROOT_DIR, folder)
    if os.path.isdir(candidate):
        MODEL_DIR = candidate
        break

if MODEL_DIR is None:
    raise FileNotFoundError(
        "Could not find a model or models folder in the project root."
    )


def model_file(filename, required=True):
    path = os.path.join(MODEL_DIR, filename)
    if required and not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing model file: {path}\n"
            "Check that your trained model files are in model/ or models/."
        )
    return path


# --------------------------------------------------
# LOAD YOUR EXISTING TRAINED MODELS
# --------------------------------------------------
clf = joblib.load(model_file("asl_model.pkl"))
scaler = joblib.load(model_file("autoencoder_scaler.pkl"))
threshold = float(joblib.load(model_file("autoencoder_threshold.pkl")))

autoencoder = AutoEncoder()
autoencoder.load_state_dict(
    torch.load(model_file("autoencoder.pt"), map_location="cpu")
)
autoencoder.eval()

motion_model = None
motion_classes = []
SEQ_LEN = 30

motion_path = model_file("motion_model.pt", required=False)

if os.path.isfile(motion_path):
    checkpoint = torch.load(motion_path, map_location="cpu")
    motion_classes = checkpoint["classes"]
    SEQ_LEN = checkpoint["seq_len"]

    motion_model = MotionNet(
        len(motion_classes),
        checkpoint["cell"],
        checkpoint["bidirectional"]
    )
    motion_model.load_state_dict(checkpoint["state_dict"])
    motion_model.eval()


# --------------------------------------------------
# SHARED APPLICATION STATE
# --------------------------------------------------
state_lock = threading.RLock()
frame_lock = threading.Lock()

state = {
    "caption": "",
    "current_word": "",
    "suggestions": [],
    "similar": [],
    "detection": "Waiting for hand...",
    "mode": "STATIC",
    "status": "Starting camera...",
    "speak_id": 0
}

latest_frame = None
camera_running = True
seq_buffer = deque(maxlen=SEQ_LEN)


def visible_caption():
    with state_lock:
        return state["caption"] + state["current_word"]


def refresh_suggestions():
    with state_lock:
        word = state["current_word"]
        previous_words = state["caption"].split()
        previous_word = previous_words[-1] if previous_words else None

    if not word or suggest_words is None:
        suggestions = []
    else:
        try:
            suggestions = suggest_words(word, previous_word) or []
        except Exception:
            suggestions = []

    with state_lock:
        state["suggestions"] = list(suggestions)[:5]


def commit(token):
    """Add a recognised sign to the caption."""
    token = str(token).upper().strip()

    with state_lock:
        if token in ("UNKNOWN", "NOTHING", ""):
            return

        if token in ("SPACE", "SP"):
            if state["current_word"]:
                state["caption"] += state["current_word"]
                state["current_word"] = ""

            if state["caption"] and not state["caption"].endswith(" "):
                state["caption"] += " "

        elif token in ("DEL", "DELETE", "BACKSPACE"):
            if state["current_word"]:
                state["current_word"] = state["current_word"][:-1]
            else:
                state["caption"] = state["caption"][:-1]

        elif token in ("SPEAK", "SPOKEN"):
            state["speak_id"] += 1

        elif len(token) == 1 and token.isalpha():
            state["current_word"] += token

        else:
            # Word-level dynamic gestures such as HELLO or THANKYOU.
            if state["current_word"]:
                state["caption"] += state["current_word"] + " "
                state["current_word"] = ""

            state["caption"] += token + " "

        state["caption"] = state["caption"]

    refresh_suggestions()


def handle_action(action):
    action = str(action).lower()

    if action == "space":
        commit("SPACE")

    elif action == "backspace":
        commit("DEL")

    elif action == "clear":
        with state_lock:
            state["caption"] = ""
            state["current_word"] = ""
            state["suggestions"] = []
            state["similar"] = []

    elif action == "speak":
        # The browser will read the current caption aloud.
        with state_lock:
            state["speak_id"] += 1

    elif action == "toggle_mode":
        with state_lock:
            state["mode"] = (
                "MOTION" if state["mode"] == "STATIC" else "STATIC"
            )
            state["detection"] = "Mode changed - show a gesture"

        seq_buffer.clear()


# --------------------------------------------------
# CAMERA AND RECOGNITION LOOP
# --------------------------------------------------
def camera_worker():
    global latest_frame

    hands_module = mp.solutions.hands
    drawing = mp.solutions.drawing_utils

    hands = hands_module.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        with state_lock:
            state["status"] = (
                "Camera unavailable. Close other apps using the webcam "
                "and restart."
            )
        hands.close()
        return

    with state_lock:
        state["status"] = "Camera connected"

    candidate = ""
    candidate_since = time.time()
    last_committed = ""
    last_motion_time = 0.0
    last_detection = "No hand detected"

    while camera_running:
        ok, frame = camera.read()

        if not ok:
            with state_lock:
                state["status"] = "Unable to read camera frame"
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        row = None

        if result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0]
            drawing.draw_landmarks(
                frame, landmarks, hands_module.HAND_CONNECTIONS
            )
            row = np.array(
                [[point.x, point.y, point.z]
                 for point in landmarks.landmark],
                dtype=np.float32
            ).flatten()

        with state_lock:
            mode = state["mode"]

        if mode == "STATIC":
            if row is None:
                candidate = ""
                last_committed = ""
                last_detection = "No hand detected"

                with state_lock:
                    state["detection"] = last_detection
                    state["similar"] = []

            else:
                row2 = normalize(row.reshape(1, -1))

                with torch.no_grad():
                    model_input = torch.tensor(
                        scaler.transform(row2).astype(np.float32)
                    )
                    reconstruction = autoencoder(model_input)
                    error = torch.mean(
                        (reconstruction - model_input) ** 2
                    ).item()

                if error > threshold:
                    prediction = "UNKNOWN"
                else:
                    prediction = str(clf.predict(row2)[0]).upper()

                last_detection = prediction

                with state_lock:
                    state["detection"] = prediction

                if len(prediction) == 1 and recommend_similar_signs:
                    try:
                        with state_lock:
                            state["similar"] = list(
                                recommend_similar_signs(prediction) or []
                            )[:3]
                    except Exception:
                        pass

                now = time.time()

                if prediction != candidate:
                    candidate = prediction
                    candidate_since = now

                # A held gesture is entered once, not repeatedly.
                if (
                    prediction not in ("UNKNOWN", "NOTHING")
                    and prediction != last_committed
                    and now - candidate_since >= 0.8
                ):
                    commit(prediction)
                    last_committed = prediction

        else:
            seq_buffer.append(
                row if row is not None
                else np.zeros(63, dtype=np.float32)
            )

            if motion_model is None:
                last_detection = "Motion model not found"
            elif len(seq_buffer) == SEQ_LEN and row is not None:
                now = time.time()

                if now - last_motion_time >= 1.5:
                    sequence = np.asarray(
                        seq_buffer, dtype=np.float32
                    ).reshape(1, SEQ_LEN, 63)

                    with torch.no_grad():
                        output = motion_model(
                            torch.tensor(sequence, dtype=torch.float32)
                        )
                        probabilities = torch.softmax(output, dim=1)[0]
                        confidence, index = torch.max(probabilities, dim=0)

                    label = motion_classes[int(index)]
                    confidence = float(confidence)

                    last_detection = (
                        f"{label} ({confidence * 100:.0f}%)"
                    )

                    if confidence >= 0.90 and label.upper() != "NOTHING":
                        if label.upper() in ("SPEAK", "SPOKEN"):
                            with state_lock:
                                state["speak_id"] += 1
                        else:
                            commit(label)

                        seq_buffer.clear()
                        last_motion_time = now

            with state_lock:
                state["detection"] = last_detection

        # Draw a simple status overlay on the camera stream.
        with state_lock:
            mode_text = state["mode"]
            caption_text = (
                state["caption"] + state["current_word"]
            )

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (25, 35, 45), -1)
        cv2.putText(
            frame, f"ASL MODE: {mode_text}", (15, 27),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
        )
        cv2.putText(
            frame, f"Sign: {last_detection[:35]}", (15, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 235, 210), 1
        )

        with frame_lock:
            latest_frame = frame.copy()

    camera.release()
    hands.close()


def video_stream():
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is None:
            time.sleep(0.1)
            continue

        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            encoded.tobytes() + b"\r\n"
        )


# --------------------------------------------------
# WEB ROUTES
# --------------------------------------------------
@app.route("/")
def home():
    return render_template_string(PAGE)


@app.route("/video_feed")
def video_feed():
    return Response(
        video_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/state")
def get_state():
    with state_lock:
        return jsonify({
            **state,
            "caption_display": state["caption"] + state["current_word"]
        })


@app.route("/api/action", methods=["POST"])
def api_action():
    payload = request.get_json(silent=True) or {}
    action = payload.get("action", "")
    allowed = {
        "space", "backspace", "clear", "speak", "toggle_mode"
    }

    if action not in allowed:
        return jsonify({"error": "Unknown action"}), 400

    handle_action(action)
    return jsonify({"ok": True})


# --------------------------------------------------
# ATTRACTIVE, RESPONSIVE BROWSER INTERFACE
# --------------------------------------------------
PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SignSpeak | ASL Recognition</title>
<style>
:root {
  --navy: #172b3a;
  --teal: #267c78;
  --mint: #e6f3ed;
  --paper: #f6f8f5;
  --ink: #23343c;
  --muted: #6b7b80;
  --line: #dce6e1;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Segoe UI", Arial, sans-serif;
  background: var(--paper);
  color: var(--ink);
}
header {
  background: var(--navy);
  color: white;
  padding: 18px 6%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 15px;
}
.brand { font-size: 22px; font-weight: 700; letter-spacing: .3px; }
.brand span { color: #a9dfc9; }
.header-note { font-size: 13px; color: #d3e4df; }
main { max-width: 1250px; margin: 28px auto; padding: 0 22px 35px; }
.intro { margin-bottom: 22px; }
.intro h1 { margin: 0 0 8px; font-size: clamp(26px, 4vw, 36px); }
.intro p { margin: 0; color: var(--muted); line-height: 1.6; }
.layout {
  display: grid;
  grid-template-columns: minmax(0, 1.15fr) minmax(310px, .85fr);
  gap: 22px;
  align-items: start;
}
.card {
  background: white;
  border: 1px solid var(--line);
  border-radius: 18px;
  overflow: hidden;
  box-shadow: 0 6px 24px rgba(23,43,58,.045);
}
.card-head {
  padding: 17px 20px;
  border-bottom: 1px solid var(--line);
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
}
.card-head h2 { font-size: 17px; margin: 0; }
.pill {
  background: var(--mint);
  color: #226b5c;
  padding: 6px 10px;
  border-radius: 30px;
  font-size: 12px;
  font-weight: 700;
}
.camera-wrap { background: #101b22; padding: 10px; }
.camera-wrap img { display: block; width: 100%; min-height: 250px; object-fit: contain; border-radius: 10px; }
.status { padding: 12px 20px; color: var(--muted); font-size: 13px; }
.content { padding: 20px; }
.label {
  text-transform: uppercase;
  font-size: 11px;
  letter-spacing: 1.1px;
  color: var(--muted);
  font-weight: 700;
  margin-bottom: 10px;
}
.caption {
  min-height: 145px;
  background: #f9fbf9;
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 17px;
  font-size: 25px;
  line-height: 1.5;
  overflow-wrap: anywhere;
  white-space: pre-wrap;
}
.placeholder { color: #9aa8a3; font-size: 18px; }
.button-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  margin-top: 15px;
}
button {
  font: inherit;
  font-weight: 650;
  border: 1px solid var(--line);
  background: white;
  color: var(--ink);
  border-radius: 10px;
  padding: 13px 12px;
  cursor: pointer;
  transition: transform .15s, background .15s;
}
button:hover { transform: translateY(-1px); background: #f0f7f3; }
button.primary { background: var(--teal); color: white; border-color: var(--teal); }
button.primary:hover { background: #1d6965; }
button.danger { color: #a34343; }
.full { width: 100%; margin-top: 10px; }
.small-section { margin-top: 22px; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chip {
  background: #eff6f1; color: #315b4f;
  border: 1px solid #d8e9de; padding: 8px 11px;
  border-radius: 9px; font-size: 13px;
}
.help {
  background: #eef5f2; border-radius: 12px; padding: 14px 16px;
  margin-top: 18px; font-size: 13px; line-height: 1.7; color: #4b6660;
}
footer { text-align: center; padding: 15px; font-size: 12px; color: var(--muted); }
@media(max-width: 800px) {
  .layout { grid-template-columns: 1fr; }
  header { padding: 16px 20px; }
  .header-note { display: none; }
  main { margin-top: 20px; padding: 0 14px 25px; }
}
</style>
</head>
<body>
<header>
  <div class="brand">Sign<span>Speak</span> ✋</div>
  <div class="header-note">American Sign Language · Recognition & Communication</div>
</header>

<main>
  <section class="intro">
    <h1>Turn gestures into words.</h1>
    <p>Show one hand to the camera, build your message, and use speech to communicate.</p>
  </section>

  <div class="layout">
    <section class="card">
      <div class="card-head">
        <h2>Live gesture recognition</h2>
        <span class="pill" id="modePill">STATIC MODE</span>
      </div>
      <div class="camera-wrap">
        <img src="/video_feed" alt="Live webcam with hand landmarks">
      </div>
      <div class="status">
        <strong>Detected:</strong> <span id="detection">Waiting for hand...</span>
        <br><span id="statusText">Connecting to camera...</span>
      </div>
      <div class="content">
        <button id="modeButton" class="full" onclick="action('toggle_mode')">
          Switch to Motion Mode
        </button>
        <div class="help">
          <strong>How to use:</strong><br>
          1. Keep your hand clearly visible in the camera.<br>
          2. Hold a letter sign steady briefly for recognition.<br>
          3. Use the buttons to edit the caption and speak your message.
        </div>
      </div>
    </section>

    <section class="card">
      <div class="card-head">
        <h2>Your message</h2>
        <span class="pill">CAPTION</span>
      </div>
      <div class="content">
        <div class="label">Recognized text</div>
        <div id="caption" class="caption"><span class="placeholder">Your message will appear here…</span></div>

        <div class="button-grid">
          <button class="primary" onclick="speakCaption()">🔊 Speak</button>
          <button onclick="action('space')">␣ Space</button>
          <button onclick="action('backspace')">⌫ Backspace</button>
          <button class="danger" onclick="clearCaption()">✕ Clear All</button>
        </div>

        <div class="small-section">
          <div class="label">Word suggestions</div>
          <div class="chips" id="suggestions"><span class="placeholder">Suggestions appear as you type.</span></div>
        </div>

        <div class="small-section">
          <div class="label">Similar signs</div>
          <div class="chips" id="similar"><span class="placeholder">Related signs appear here.</span></div>
        </div>

        <div class="help">
          <strong>Accessibility note:</strong> The Speak button uses your browser's
          text-to-speech voice. Available voices depend on your device and browser.
        </div>
      </div>
    </section>
  </div>
</main>
<footer>SignSpeak · ASL Recognition Project</footer>

<script>
let lastSpeakId = 0;

async function action(name) {
  try {
    await fetch('/api/action', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action: name})
    });
    await refreshState();
  } catch (error) {
    document.getElementById('statusText').textContent =
      'Connection issue. Refresh the page or restart the application.';
  }
}

async function refreshState() {
  try {
    const response = await fetch('/api/state');
    const data = await response.json();

    const caption = document.getElementById('caption');
    caption.textContent = data.caption_display || '';
    if (!data.caption_display) {
      caption.innerHTML = '<span class="placeholder">Your message will appear here…</span>';
    }

    document.getElementById('detection').textContent = data.detection;
    document.getElementById('statusText').textContent = data.status;
    document.getElementById('modePill').textContent = data.mode + ' MODE';
    document.getElementById('modeButton').textContent =
      data.mode === 'STATIC' ? 'Switch to Motion Mode' : 'Switch to Static Mode';

    renderChips('suggestions', data.suggestions, 'Suggestions appear as you type.');
    renderChips('similar', data.similar, 'Related signs appear here.');

    if (data.speak_id !== lastSpeakId) {
      lastSpeakId = data.speak_id;
      speakText(data.caption_display);
    }
  } catch (error) {
    document.getElementById('statusText').textContent =
      'Cannot connect to the Python application.';
  }
}

function renderChips(elementId, values, emptyMessage) {
  const element = document.getElementById(elementId);
  element.replaceChildren();

  if (!values || values.length === 0) {
    const span = document.createElement('span');
    span.className = 'placeholder';
    span.textContent = emptyMessage;
    element.appendChild(span);
    return;
  }

  values.forEach(value => {
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.textContent = typeof value === 'string'
      ? value
      : JSON.stringify(value);
    element.appendChild(chip);
  });
}

function speakText(text) {
  if (!text || !text.trim()) {
    document.getElementById('statusText').textContent =
      'Add a letter or word before speaking.';
    return;
  }

  if (!('speechSynthesis' in window)) {
    document.getElementById('statusText').textContent =
      'Text-to-speech is not supported in this browser.';
    return;
  }

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text.trim());
  utterance.lang = 'en-US';
  window.speechSynthesis.speak(utterance);
}

async function speakCaption() {
  try {
    const response = await fetch('/api/state');
    const data = await response.json();
    speakText(data.caption_display);
  } catch (error) {
    document.getElementById('statusText').textContent =
      'Unable to read the caption.';
  }
}

async function clearCaption() {
  window.speechSynthesis.cancel();
  await action('clear');
}

setInterval(refreshState, 400);
refreshState();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    worker = threading.Thread(target=camera_worker, daemon=True)
    worker.start()

    print("\nSignSpeak is starting.")
    print("Open this address in your browser: http://127.0.0.1:5000")
    print("Keep this terminal open while using the application.")
    try:
        app.run(
            host="127.0.0.1",
            port=5000,
            debug=False,
            threaded=True,
            use_reloader=False
        )
    finally:
        camera_running = False
