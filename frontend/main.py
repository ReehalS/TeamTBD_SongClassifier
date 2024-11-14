import streamlit as st
import pandas as pd
import librosa
import numpy as np
import tempfile
import os

# function to extract audio features
def extract_features(file_path):
    y, sr = librosa.load(file_path)
    features = {
        "length": librosa.get_duration(y=y, sr=sr),
        "chroma_stft_mean": np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
        "chroma_stft_var": np.var(librosa.feature.chroma_stft(y=y, sr=sr)),
        "rms_mean": np.mean(librosa.feature.rms(y=y)),
        "rms_var": np.var(librosa.feature.rms(y=y)),
        "spectral_centroid_mean": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        "spectral_centroid_var": np.var(librosa.feature.spectral_centroid(y=y, sr=sr)),
        "spectral_bandwidth_mean": np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        "spectral_bandwidth_var": np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        "rolloff_mean": np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        "rolloff_var": np.var(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        "zero_crossing_rate_mean": np.mean(librosa.feature.zero_crossing_rate(y)),
        "zero_crossing_rate_var": np.var(librosa.feature.zero_crossing_rate(y)),
        "harmony_mean": np.mean(librosa.effects.harmonic(y)),
        "harmony_var": np.var(librosa.effects.harmonic(y)),
        "perceptr_mean": np.mean(librosa.effects.percussive(y)),
        "perceptr_var": np.var(librosa.effects.percussive(y)),
        "tempo": librosa.beat.tempo(y=y, sr=sr)[0]
    }
    
    # Extract MFCCs and add mean and variance of each MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(1, 21):
        features[f"mfcc{i}_mean"] = np.mean(mfccs[i - 1])
        features[f"mfcc{i}_var"] = np.var(mfccs[i - 1])
    
    return features

st.title("Team _TBD_", anchor=False)
st.header("Song Genre Classifier :musical_note:", False)
st.subheader("Enter a .mp3 file: ", False)

uploaded_file = st.file_uploader("Choose a file", type=["mp3"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    
    # Extract features from the MP3 file
    try:
        features = extract_features(temp_file_path)
        # Display the features in a table
        st.write("Extracted Features:")
        st.write(pd.DataFrame([features]))
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
    finally:
        # Clean up temporary file
        os.remove(temp_file_path)

    