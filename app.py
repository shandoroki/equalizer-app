import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import firwin, lfilter
import io
import librosa
import matplotlib.pyplot as plt

# --- Apply custom styles ---
st.markdown("""
    <style>
    .stApp {
        background-color: #001f3f;
        color: white;
    }
    .stSlider > div[data-baseweb="slider"] > div {
        background: #00ffff !important;
        height: 8px;
    }
    .stSlider > label {
        color: #00ffff;
        font-weight: bold;
    }
    .stDownloadButton > button {
        background-color: #0074D9;
        color: white;
        font-weight: bold;
    }
    h1, h2, h3, .st-bb, .st-c2 {
        color: #00ffff;
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
        st.download_button("⬇️ Download Processed Audio", buf.getvalue(), file_name="equalized_output.wav")

        # --- Visualization with matplotlib ---
        st.subheader("📈 Waveform Visualization")
        fig, ax = plt.subplots(figsize=(10, 3))
        time = np.linspace(0, len(output) / fs, num=len(output))
        ax.plot(time, output, color="#00ffff", linewidth=0.8)
        ax.set_facecolor("black")
        fig.patch.set_facecolor("black")
        ax.set_xlabel("Time [s]", fontsize=10, fontweight='bold', color='white')
        ax.set_ylabel("Amplitude", fontsize=10, fontweight='bold', color='white')
        ax.set_title("Processed Audio Waveform", fontsize=12, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        st.pyplot(fig)
