import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import firwin, lfilter
import io
import librosa
import matplotlib.pyplot as plt

# --- DJ-Themed Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&display=swap');

    .stApp {
        background: linear-gradient(145deg, #0d0d0d, #1a1a1a);
        color: #f0f0f0;
        font-family: 'Orbitron', sans-serif;
    }

    h1, h2, h3 {
        color: #00ffe1;
        text-shadow: 0 0 10px #00ffe1;
    }

    .stSlider > div {
        background-color: #111;
        border-radius: 10px;
        padding: 0.5em;
    }

    .stSlider input[type=range]::-webkit-slider-thumb {
        background: #00ffe1;
        box-shadow: 0 0 10px #00ffe1;
    }

    .stSlider input[type=range]::-webkit-slider-runnable-track {
        background: #444;
    }

    .css-1cpxqw2, .css-14r9z6v {
        background-color: #111 !important;
    }

    .css-1cpxqw2:hover {
        background-color: #00ffe1 !important;
        color: black !important;
    }

    audio {
        filter: drop-shadow(0 0 10px #00ffe1aa);
    }

    .stDownloadButton button {
        background: #00ffe1;
        color: black;
        border: none;
        font-weight: bold;
        border-radius: 10px;
        box-shadow: 0 0 10px #00ffe1;
    }

    .stDownloadButton button:hover {
        background: #0ff;
        color: #000;
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
st.title("🎛️ Digital Music Equalizer")

uploaded_file = st.file_uploader("Upload your audio track (WAV or MP3)", type=["wav", "mp3"])

if uploaded_file is not None:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > 100:
        st.error("⚠️ File size exceeds 100 MB limit. Please upload a smaller file.")
    else:
        data, fs = load_audio(uploaded_file)
        st.audio(uploaded_file)

        st.subheader("🎚️ Tweak Your Frequencies")
        bass = st.slider("Bass Boost (60–250 Hz)", 0.0, 2.0, 1.0, 0.1)
        mid = st.slider("Midrange Boost (250 Hz – 4 kHz)", 0.0, 2.0, 1.0, 0.1)
        treble = st.slider("Treble Boost (4–10 kHz)", 0.0, 2.0, 1.0, 0.1)

        output = apply_equalizer(data, fs, [bass, mid, treble])

        # Playback + Download
        buf = io.BytesIO()
        sf.write(buf, output, fs, format='WAV')
        st.audio(buf, format='audio/wav')
        st.download_button("⬇️ Download DJ-Processed Track", buf.getvalue(), file_name="dj_equalized_output.wav")

        # --- Visualizer: Original vs Processed ---
        st.subheader("🔊 Audio Waveform Visualizer")
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        time = np.linspace(0, len(data) / fs, num=len(data))

        axs[0].plot(time, data, color="#888", linewidth=0.5)
        axs[0].set_title("Original Track", fontsize=12, color='#00ffe1')
        axs[0].set_ylabel("Amplitude", color='white')
        axs[0].set_facecolor("#111")
        axs[0].tick_params(colors='white')

        axs[1].plot(time, output, color="#00ffe1", linewidth=0.5)
        axs[1].set_title("Processed Track", fontsize=12, color='#00ffe1')
        axs[1].set_xlabel("Time [s]", color='white')
        axs[1].set_ylabel("Amplitude", color='white')
        axs[1].set_facecolor("#111")
        axs[1].tick_params(colors='white')

        fig.patch.set_facecolor("#111")
        st.pyplot(fig)
