import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import firwin, lfilter
import io
import librosa
import matplotlib.pyplot as plt

# --- Hot Pink DJ-Themed Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&display=swap');

    .stApp {
        background-color: #000000;
        color: #ff69b4;
        font-family: 'Orbitron', sans-serif;
    }

    h1, h2, h3 {
        color: #ff69b4;
        text-shadow: 0 0 10px #ff69b4;
    }

    .stSlider > div {
        background-color: #111;
        border-radius: 10px;
        padding: 0.5em;
    }

    .stSlider input[type=range]::-webkit-slider-thumb {
        background: #ff69b4;
        box-shadow: 0 0 10px #ff69b4;
    }

    .stSlider input[type=range]::-webkit-slider-runnable-track {
        background: #333;
    }

    .css-1cpxqw2, .css-14r9z6v {
        background-color: #111 !important;
    }

    .css-1cpxqw2:hover {
        background-color: #ff69b4 !important;
        color: black !important;
    }

    audio {
        filter: drop-shadow(0 0 10px #ff69b4aa);
    }

    .stDownloadButton button {
        background: #ff69b4;
        color: black;
        border: none;
        font-weight: bold;
        border-radius: 10px;
        box-shadow: 0 0 10px #ff69b4;
    }

    .stDownloadButton button:hover {
        background: #ff85c1;
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

# --- Audio Functions ---
def load_audio(file):
    y, sr = librosa.load(file, sr=None, mono=True)
    return y, sr

def bandpass_filter(data, lowcut, highcut, fs, numtaps=101):
    taps = firwin(numtaps, [lowcut, highcut], pass_zero=False, fs=fs)
    return lfilter(taps, 1.0, data)

def apply_equalizer(data, fs, gains):
    bands = [(60, 250), (250, 4000), (4000, 10000)]  # Bass, Mid, Treble
    processed = np.zeros_like(data)
    for (low, high), gain in zip(bands, gains):
        filtered = bandpass_filter(data, low, high, fs)
        processed += filtered * gain
    return processed

# --- UI ---
st.title("🎛️ Hot Pink DJ Equalizer")

uploaded_file = st.file_uploader("🎵 Upload your audio track (WAV or MP3)", type=["wav", "mp3"])

if uploaded_file is not None:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > 100:
        st.error("⚠️ File size exceeds 100 MB limit. Please upload a smaller file.")
    else:
        data, fs = load_audio(uploaded_file)
        st.audio(uploaded_file)

        st.subheader("🎚️ Adjust the Vibes")
        bass = st.slider("Bass Boost (60–250 Hz)", 0.0, 2.0, 1.0, 0.1)
        mid = st.slider("Midrange Boost (250 Hz – 4 kHz)", 0.0, 2.0, 1.0, 0.1)
        treble = st.slider("Treble Boost (4–10 kHz)", 0.0, 2.0, 1.0, 0.1)

        output = apply_equalizer(data, fs, [bass, mid, treble])

        # Playback + Download
        buf = io.BytesIO()
        sf.write(buf, output, fs, format='WAV')
        st.audio(buf, format='audio/wav')
        st.download_button("⬇️ Download Equalized Track", buf.getvalue(), file_name="hotpink_equalizer_output.wav")

        # --- Visualizer: Original vs Pro
