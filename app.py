import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import firwin, lfilter
import io
import librosa
import matplotlib.pyplot as plt

# --- Custom CSS for Dark Blue Theme ---
st.markdown("""
    <style>
        body {
            background-color: #0B1F3A;
            color: white;
        }
        .stApp {
            background-color: #0B1F3A;
        }
        .css-1cpxqw2 {  /* Slider track */
            background-color: #1F3B73 !important;
        }
        .css-14r9z6v {  /* Slider handle */
            background-color: #3A74C9 !important;
        }
        .css-1cpxqw2:hover {
            background-color: #265DAB !important;
        }
        .stSlider > div {
            padding: 1em 0.5em;
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

uploaded_file = st.file_uploader("Upload audio file (WAV or MP3)", type=["wav", "mp3"])

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
        st.subheader("📊 Original vs Processed Waveform")
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        time = np.linspace(0, len(data) / fs, num=len(data))

        # Original waveform
        axs[0].plot(time, data, color="gray", linewidth=0.5)
        axs[0].set_title("Original Audio", fontsize=12, fontweight='bold', color='white')
        axs[0].set_ylabel("Amplitude", fontsize=10, color='white')
        axs[0].set_facecolor("#0B1F3A")
        axs[0].tick_params(colors='white')

        # Processed waveform
        axs[1].plot(time, output, color="white", linewidth=0.5)
        axs[1].set_title("Processed Audio", fontsize=12, fontweight='bold', color='white')
        axs[1].set_xlabel("Time [s]", fontsize=10, color='white')
        axs[1].set_ylabel("Amplitude", fontsize=10, color='white')
        axs[1].set_facecolor("#0B1F3A")
        axs[1].tick_params(colors='white')

        fig.patch.set_facecolor("#0B1F3A")
        st.pyplot(fig)
