import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import firwin, lfilter
import io
import librosa
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# --- Custom CSS for vertical sliders and dark theme ---
st.markdown("""
    <style>
    body {
        background-color: #0b1a2d;
        color: white;
    }
    .block-container {
        background-color: #0b1a2d;
    }
    .slider-container {
        display: flex;
        justify-content: center;
        gap: 50px;
        padding: 20px;
    }
    .slider {
        -webkit-appearance: slider-vertical;
        writing-mode: bt-lr; /* bottom to top */
        width: 8px;
        height: 200px;
        background: #0b1a2d;
        accent-color: #00ffcc;
    }
    .slider-label {
        text-align: center;
        margin-top: 10px;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>🎛️ Digital Music Equalizer</h1>", unsafe_allow_html=True)

# --- Upload audio ---
uploaded_file = st.file_uploader("🎵 Upload audio file (WAV or MP3)", type=["wav", "mp3"])

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

if uploaded_file:
    data, fs = load_audio(uploaded_file)
    st.audio(uploaded_file)

    st.markdown("### 🎚️ Equalizer Controls (Vertical Sliders)")

    # Create vertical sliders using raw HTML
    bass_gain = st.slider("Bass (60–250 Hz)", 0.0, 2.0, 1.0, 0.1, key="bass")
    mid_gain = st.slider("Midrange (250–4k Hz)", 0.0, 2.0, 1.0, 0.1, key="mid")
    treble_gain = st.slider("Treble (4–10k Hz)", 0.0, 2.0, 1.0, 0.1, key="treble")

    # Simulated vertical sliders
    components.html(f"""
        <div class="slider-container">
            <div>
                <input type="range" min="0" max="2" value="{bass_gain}" step="0.1" class="slider" disabled>
                <div class="slider-label">Bass</div>
            </div>
            <div>
                <input type="range" min="0" max="2" value="{mid_gain}" step="0.1" class="slider" disabled>
                <div class="slider-label">Mid</div>
            </div>
            <div>
                <input type="range" min="0" max="2" value="{treble_gain}" step="0.1" class="slider" disabled>
                <div class="slider-label">Treble</div>
            </div>
        </div>
    """, height=300)

    output = apply_equalizer(data, fs, [bass_gain, mid_gain, treble_gain])

    buf = io.BytesIO()
    sf.write(buf, output, fs, format='WAV')
    st.audio(buf, format='audio/wav')
    st.download_button("⬇️ Download Processed Audio", buf.getvalue(), file_name="equalized_output.wav")

    st.markdown("### 📈 Processed Audio Waveform")
    fig, ax = plt.subplots(figsize=(10, 3))
    time = np.linspace(0, len(output) / fs, num=len(output))
    ax.plot(time, output, color="cyan", linewidth=0.5)
    ax.set_facecolor("#0b1a2d")
    fig.patch.set_facecolor("#0b1a2d")
    ax.set_xlabel("Time [s]", color="white")
    ax.set_ylabel("Amplitude", color="white")
    ax.set_title("Processed Audio", color="white")
    ax.tick_params(colors='white')
    st.pyplot(fig)
