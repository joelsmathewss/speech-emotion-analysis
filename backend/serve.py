import os
import uuid
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "trained_model.h5")

EMOTION_LABELS = ["neutral","calm","happy","sad","angry","fearful","disgust","surprised"]
EMOTION_EMOJIS = {
    "neutral":"😐","calm":"😌","happy":"😊","sad":"😢",
    "angry":"😠","fearful":"😨","disgust":"🤢","surprised":"😲",
}

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

print("Loading model...", end=" ", flush=True)
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✓")
except Exception as e:
    model = None
    print(f"✗  ({e})")
    print("  → Run `python sample.py` first to train the model.")

def extract_features(file_path, n_mfcc=40):
    y, sr  = librosa.load(file_path, sr=None, mono=True)
    mfcc   = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel    = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    features = np.concatenate([mfcc, chroma, mel])
    return features[np.newaxis, ..., np.newaxis]   # shape: (1, N, 1)

@app.get("/hello")
def hello():
    return jsonify({"message": "Speech Emotion API is running ✓"})

@app.post("/predict")
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run sample.py first."}), 503
    if "audio" not in request.files:
        return jsonify({"error": "No audio file. Send it with key 'audio'."}), 400
    file = request.files["audio"]
    ext  = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in {"wav", "mp3", "flac"}:
        return jsonify({"error": "Unsupported format. Use WAV, MP3, or FLAC."}), 400
    tmp = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}.{ext}")
    try:
        file.save(tmp)
        features    = extract_features(tmp)
        preds       = model.predict(features, verbose=0)[0]
        idx         = int(np.argmax(preds))
        emotion     = EMOTION_LABELS[idx]
        emoji       = EMOTION_EMOJIS.get(emotion, "")
        confidence  = round(float(preds[idx]) * 100, 1)
        return jsonify({
            "predicted_emotion": f"{emotion.capitalize()} {emoji}",
            "emotion":    emotion,
            "emoji":      emoji,
            "confidence": confidence,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)