import { useState, useRef } from "react";

const API = "http://127.0.0.1:5000";

const EMOTIONS = {
  neutral:   { emoji: "😐", bg: "#e5e7eb", accent: "#6b7280", label: "Neutral"   },
  calm:      { emoji: "😌", bg: "#dbeafe", accent: "#3b82f6", label: "Calm"      },
  happy:     { emoji: "😊", bg: "#fef9c3", accent: "#eab308", label: "Happy"     },
  sad:       { emoji: "😢", bg: "#bfdbfe", accent: "#1d4ed8", label: "Sad"       },
  angry:     { emoji: "😠", bg: "#fee2e2", accent: "#ef4444", label: "Angry"     },
  fearful:   { emoji: "😨", bg: "#ede9fe", accent: "#7c3aed", label: "Fearful"   },
  disgust:   { emoji: "🤢", bg: "#dcfce7", accent: "#16a34a", label: "Disgust"   },
  surprised: { emoji: "😲", bg: "#fce7f3", accent: "#ec4899", label: "Surprised" },
};

function beep() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.frequency.value = 880;
    gain.gain.setValueAtTime(0.1, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.15);
    osc.start();
    osc.stop(ctx.currentTime + 0.15);
  } catch (_) {}
}

export default function App() {
  const [result, setResult]       = useState(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);
  const [recording, setRecording] = useState(false);
  const [tab, setTab]             = useState("upload");

  const mediaRef     = useRef(null);
  const chunksRef    = useRef([]);
  const fileInputRef = useRef(null);

  const emotion = result ? EMOTIONS[result.emotion] ?? null : null;
  const bgColor = emotion ? emotion.bg : "#f9fafb";

  async function handleFile(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const form = new FormData();
      form.append("audio", file);
      const res  = await fetch(`${API}/predict`, { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Prediction failed");
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      e.target.value = "";
    }
  }

  async function startRecording() {
    setError(null);
    setResult(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      chunksRef.current = [];
      const mr = new MediaRecorder(stream);
      mr.ondataavailable = (e) => chunksRef.current.push(e.data);
      mr.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        beep();
        const blob = new Blob(chunksRef.current, { type: "audio/wav" });
        setLoading(true);
        try {
          const form = new FormData();
          form.append("audio", blob, "recording.wav");
          const res  = await fetch(`${API}/predict`, { method: "POST", body: form });
          const data = await res.json();
          if (!res.ok) throw new Error(data.error || "Prediction failed");
          setResult(data);
        } catch (err) {
          setError(err.message);
        } finally {
          setLoading(false);
        }
      };
      mr.start();
      mediaRef.current = mr;
      setRecording(true);
      beep();
    } catch {
      setError("Microphone access denied. Please allow it in browser settings.");
    }
  }

  function stopRecording() {
    mediaRef.current?.stop();
    setRecording(false);
  }

  function reset() {
    setResult(null);
    setError(null);
  }

  return (
    <div
      className="min-h-screen flex flex-col items-center justify-center px-4 py-12 transition-all duration-700"
      style={{ backgroundColor: bgColor }}
    >
      {/* Header */}
      <header className="mb-10 text-center">
        <p className="font-mono-dm text-xs tracking-widest text-gray-400 uppercase mb-2">
          Deep Learning · Audio Analysis
        </p>
        <h1 className="text-3xl font-semibold text-gray-800 tracking-tight">
          Speech Emotion Detection
        </h1>
      </header>

      {/* Card */}
      <div className="w-full max-w-md bg-white/80 backdrop-blur-sm rounded-2xl shadow-sm border border-white/60 p-8">

        {/* Tabs */}
        <div className="flex gap-1 bg-gray-100 rounded-lg p-1 mb-8">
          {["upload", "record"].map((t) => (
            <button
              key={t}
              onClick={() => { setTab(t); reset(); }}
              className={`flex-1 py-2 text-sm rounded-md font-medium transition-all ${
                tab === t
                  ? "bg-white text-gray-800 shadow-sm"
                  : "text-gray-500 hover:text-gray-700"
              }`}
            >
              {t === "upload" ? "↑ Upload File" : "⏺ Record Audio"}
            </button>
          ))}
        </div>

        {/* Upload */}
        {tab === "upload" && (
          <div
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed border-gray-200 rounded-xl p-10 text-center cursor-pointer hover:border-gray-400 hover:bg-gray-50 transition-all"
          >
            <p className="text-4xl mb-3">🎵</p>
            <p className="text-sm text-gray-600 mb-1">
              Drop an audio file or click to browse
            </p>
            <p className="font-mono-dm text-xs text-gray-400">WAV · MP3 · FLAC</p>
            <input
              ref={fileInputRef}
              type="file"
              accept=".wav,.mp3,.flac"
              className="hidden"
              onChange={handleFile}
            />
          </div>
        )}

        {/* Record */}
        {tab === "record" && (
          <div className="flex flex-col items-center gap-5 py-6">
            <div className="relative flex items-center justify-center">
              {recording && (
                <span
                  className="absolute w-20 h-20 rounded-full pulse-ring"
                  style={{ backgroundColor: "#ef444460" }}
                />
              )}
              <button
                onClick={recording ? stopRecording : startRecording}
                className={`relative z-10 w-16 h-16 rounded-full flex items-center justify-center text-white text-2xl transition-all shadow-md ${
                  recording
                    ? "bg-red-500 hover:bg-red-600"
                    : "bg-gray-800 hover:bg-gray-700"
                }`}
              >
                {recording ? "⏹" : "🎤"}
              </button>
            </div>
            <p className="text-sm text-gray-500">
              {recording ? "Recording… tap to stop" : "Tap to start recording"}
            </p>
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div className="mt-6 flex items-center justify-center gap-3 text-gray-500">
            <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10"
                stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
            </svg>
            <span className="text-sm">Analysing audio…</span>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="mt-6 p-4 bg-red-50 border border-red-100 rounded-xl text-sm text-red-600">
            {error}
          </div>
        )}

        {/* Result */}
        {result && emotion && !loading && (
          <div
            className="mt-6 fade-up rounded-xl p-6 text-center"
            style={{ backgroundColor: emotion.bg }}
          >
            <span className="text-6xl block mb-3">{emotion.emoji}</span>
            <p className="text-2xl font-semibold mb-1" style={{ color: emotion.accent }}>
              {emotion.label}
            </p>
            <p className="font-mono-dm text-xs text-gray-500">
              {result.confidence}% confidence
            </p>
            <button
              onClick={reset}
              className="mt-4 text-xs text-gray-400 hover:text-gray-600 underline transition-colors"
            >
              Try another
            </button>
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="mt-10 font-mono-dm text-xs text-gray-400 text-center">
        CNN · RAVDESS git 
      </footer>
    </div>
  );
}