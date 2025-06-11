import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import firwin, lfilter
import io
import librosa
import matplotlib.pyplot as plt

# --- Custom CSS for dark blue theme and cyan sliders ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0a1e3c;
        color: white;
    }
    /* Thicker, cyan sliders */
    input[type=range] {
        accent-color: #00ffff;
        height: 8px;
    }
    .css-1cpxqw2, .stSlider > div {
        background-color: #00ffff !important;
        height: 8px !important;
        border-radius: 10px !important;
    }
    .css-14pt78w, .css-1cpgz07 {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Audio loading function ---
def load_audio(file):
    y, sr = librosa.load(file, sr=None, mono=True)
    return y, sr

# --- Bandpass filter function ---
def bandpass_filter(data, lowcut, highcut, fs, numtaps=101):
    taps = firwin(numtaps, [lowcut, highcut], pass_zero=False, fs=fs)
    return lfilter(taps, 1.0, data)

# --- Apply equalizer with adjustable gains ---
def apply_equalizer(data, fs, gains):
    bands = [(60, 250), (250, 4000), (4000, 10000)]  # Bass, Mid, Treble
    processed = np.zeros_like(data)
    for (low, high), gain in zip(bands, gains):
        filtered = bandpass_filter(data, low, high, fs)
        processed += filtered * gain
    return processed

# --- Streamlit UI ---
st.title("🎙️ Digital Music Equalizer")

uploaded_file = st.file_uploader("🎵 Upload audio file (WAV or MP3)", type=["wav", "mp3"])

if uploaded_file is not None:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > 100:
        st.error("⚠️ File size exceeds 100 MB limit. Please upload a smaller file.")
    else:
        data, fs = load_audio(uploaded_file)
        st.audio(uploaded_file)

        st.subheader("🎚️ Adjust Frequency Bands")
        bass = st.slider("Bass (60–250 Hz)", 0.0, 2.0, 1.0, 0.1)
        mid = st.slider("Midrange (250 Hz – 4 kHz)", 0.0, 2.0, 1.0, 0.1)
        treble = st.slider("Treble (4–10 kHz)", 0.0, 2.0, 1.0, 0.1)

        output = apply_equalizer(data, fs, [bass, mid, treble])

        # Save to buffer and playback
        buf = io.BytesIO()
        sf.write(buf, output, fs, format='WAV')
        st.audio(buf, format='audio/wav')
        st.download_button("⬇️ Download Processed Audio", buf.getvalue(), file_n
