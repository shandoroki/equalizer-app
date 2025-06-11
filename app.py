import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import firwin, lfilter
import io
import librosa
import matplotlib.pyplot as plt

# --- Custom Styling: Hot Pink Glow + White Text ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&display=swap');

    .stApp {import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import firwin, lfilter
import io
import librosa
import matplotlib.pyplot as plt

# --- Custom Styling: Hot Pink Glow + Violet DJ Background ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0a0a0a, #1a001a);
        color: white;
        font-family: 'Orbitron', sans-serif;
    }

    h1, h2, h3 {
        color: white;
        text-shadow: 0 0 15px #ff69b4;
    }

    .stSlider > div {
        background-color: #111;
        border-radius: 10px;
        padding: 0.5em;
    }

    .stSlider input[type=range]::-webkit-slider-thumb {
        background: #ff69b4;
        box-shadow: 0 0 12px #ff69b4;
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
        font-weight: bold;
        border-radius: 10px;
        border: none;
        box-shadow: 0 0 12px #ff69b4;
    }

    .stDownloadButton button:hover {
        background: #ff85c1;
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
st.title("🎛️ DJ Equalizer: Hot Pink Glow Edition")

uploaded_file = st.file_uploader("🎵 Upload your audio track (WAV or MP3)", type=["wav", "mp3"])

if uploaded_file is not None:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > 100:
        st.error("⚠️ File size exceeds 100 MB limit. Please upload a smaller file.")
    else:
        data, fs = load_audio(uploaded_file)
        st.audio(uploaded_file)

        st.subheader("🎚️ Adjust the Frequencies")
        bass = st.slider("Bass Boost (60–250 Hz)", 0.0, 2.0, 1.0, 0.1)
        mid = st.slider("Midrange Boost (250 Hz – 4 kHz)", 0.0, 2.0, 1.0, 0.1)
        treble = st.slider("Treble Boost (4–10 kHz)", 0.0, 2.0, 1.0, 0.1)

        output = apply_equalizer(data, fs, [bass, mid, treble])

        # Save and play
        buf = io.BytesIO()
        sf.write(buf, output, fs, format='WAV')
        st.audio(buf, format='audio/wav')
        st.download_button("⬇️ Download Processed Audio", buf.getvalue(), file_name="hotpink_equalized_output.wav")

        # --- Processed Visualization Only ---
        st.subheader("🌈 Processed Track Waveform")
        fig, ax = plt.subplots(figsize=(10, 4))

        time = np.linspace(0, len(output) / fs, num=len(output))
        ax.plot(time, output, color="#ff69b4", linewidth=0.5)
        ax.set_title("Processed Audio", fontsize=12, color='#ff69b4')
        ax.set_xlabel("Time [s]", color='white')
        ax.set_ylabel("Amplitude", color='white')
        ax.set_facecolor("#0a0a0a")
        ax.tick_params(colors='white')
        fig.patch.set_facecolor("#0a0a0a")
        st.pyplot(fig)

        background-color: #000000;
        color: white;
        font-family: 'Orbitron', sans-serif;
    }

    h1, h2, h3 {
        color: white;
        text-shadow: 0 0 15px #ff69b4;
    }

    .stSlider > div {
        background-color: #111;
        border-radius: 10px;
        padding: 0.5em;
    }

    .stSlider input[type=range]::-webkit-slider-thumb {
        background: #ff69b4;
        box-shadow: 0 0 12px #ff69b4;
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
        font-weight: bold;
        border-radius: 10px;
        border: none;
        box-shadow: 0 0 12px #ff69b4;
    }

    .stDownloadButton button:hover {
        background: #ff85c1;
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

uploaded_file = st.file_uploader("🎵 Upload your audio track (WAV or MP3)", type=["wav", "mp3"])

if uploaded_file is not None:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > 100:
        st.error("⚠️ File size exceeds 100 MB limit. Please upload a smaller file.")
    else:
        data, fs = load_audio(uploaded_file)
        st.audio(uploaded_file)

        st.subheader("🎚️ Adjust the Frequencies")
        bass = st.slider("Bass Boost (60–250 Hz)", 0.0, 2.0, 1.0, 0.1)
        mid = st.slider("Midrange Boost (250 Hz – 4 kHz)", 0.0, 2.0, 1.0, 0.1)
        treble = st.slider("Treble Boost (4–10 kHz)", 0.0, 2.0, 1.0, 0.1)

        output = apply_equalizer(data, fs, [bass, mid, treble])

        # Save and play
        buf = io.BytesIO()
        sf.write(buf, output, fs, format='WAV')
        st.audio(buf, format='audio/wav')
        st.download_button("⬇️ Download Processed Audio", buf.getvalue(), file_name="hotpink_equalized_output.wav")

        # --- Visualization ---
        st.subheader("🎧 Waveform Visualizer")
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        time = np.linspace(0, len(data) / fs, num=len(data))

        axs[0].plot(time, data, color="#ff85c1", linewidth=0.5)
        axs[0].set_title("Original Track", fontsize=12, color='#ff69b4')
        axs[0].set_ylabel("Amplitude", color='white')
        axs[0].set_facecolor("#000000")
        axs[0].tick_params(colors='white')

        axs[1].plot(time, output, color="#ff69b4", linewidth=0.5)
        axs[1].set_title("Processed Track", fontsize=12, color='#ff69b4')
        axs[1].set_xlabel("Time [s]", color='white')
        axs[1].set_ylabel("Amplitude", color='white')
        axs[1].set_facecolor("#000000")
        axs[1].tick_params(colors='white')

        fig.patch.set_facecolor("#000000")
        st.pyplot(fig)
