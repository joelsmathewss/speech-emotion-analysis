import os
import glob
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

EMOTIONS = {
    "01": "neutral", "02": "calm",     "03": "happy", "04": "sad",
    "05": "angry",   "06": "fearful",  "07": "disgust","08": "surprised",
}

def extract_features(file_path, n_mfcc=40):
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
        mfcc   = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        mel    = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        return np.concatenate([mfcc, chroma, mel])
    except Exception as e:
        print(f"[WARN] Skipping {file_path}: {e}")
        return None

def load_data(data_dir="./data"):
    X, y = [], []
    pattern = os.path.join(data_dir, "Actor_*", "*.wav")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No WAV files found at '{pattern}'.\n"
            "Put the RAVDESS Actor_* folders inside backend/data/"
        )
    print(f"Found {len(files)} files. Extracting features...")
    for path in files:
        parts = os.path.basename(path).split("-")
        if len(parts) < 3:
            continue
        emotion = EMOTIONS.get(parts[2])
        if not emotion:
            continue
        features = extract_features(path)
        if features is None:
            continue
        X.append(features)
        y.append(emotion)
        # Augment with noise
        try:
            audio, sr = librosa.load(path, sr=None, mono=True)
            noise = audio + 0.005 * np.random.randn(len(audio))
            mfcc   = np.mean(librosa.feature.mfcc(y=noise, sr=sr, n_mfcc=40).T, axis=0)
            chroma = np.mean(librosa.feature.chroma_stft(y=noise, sr=sr).T, axis=0)
            mel    = np.mean(librosa.feature.melspectrogram(y=noise, sr=sr).T, axis=0)
            X.append(np.concatenate([mfcc, chroma, mel]))
            y.append(emotion)
        except Exception:
            pass
    return np.array(X), np.array(y)

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(64,  kernel_size=3, activation="relu", input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),
        Conv1D(128, kernel_size=3, activation="relu"),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def train():
    X, y_labels = load_data()
    np.save("X.npy", X)
    np.save("y.npy", y_labels)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features each")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labels)
    y_cat = to_categorical(y_encoded)
    X = X[..., np.newaxis]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Train: {len(X_train)} | Val: {len(X_val)}")
    model = build_model(X_train.shape[1:], y_cat.shape[1])
    model.summary()
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[
            EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
            ModelCheckpoint("trained_model.h5", save_best_only=True, verbose=1),
        ],
    )
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n✓ Validation accuracy: {acc:.2%}")
    print(f"✓ Model saved → trained_model.h5")
    print(f"✓ Label order: {list(le.classes_)}")

if __name__ == "__main__":
    train()