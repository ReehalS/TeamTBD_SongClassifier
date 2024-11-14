import streamlit as st
import pandas as pd
import librosa
import numpy as np
import tempfile
import os
import joblib
from collections import Counter

# Function to extract features from a 3-second audio segment
def extract_features(y, sr):
    features = {
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
st.subheader("Enter an audio file (.mp3 or .wav):", False)

uploaded_file = st.file_uploader("Choose a file", type=["mp3", "wav"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3" if uploaded_file.name.endswith('.mp3') else ".wav") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    try:
        # Load the entire audio file
        y, sr = librosa.load(temp_file_path)
        
        # Load the trained model and label encoder
        xgb_loaded = joblib.load('frontend/25p_4000_0_01_Model/xgb_model_4000_0_01_25p.pkl')
        le_loaded = joblib.load('frontend/25p_4000_0_01_Model/label_encoder_4000_0_01_25p.pkl')

        # Segment the audio into 3-second clips and make predictions on each
        segment_duration = 3  # seconds
        segment_samples = segment_duration * sr
        genre_counts = []

        for i in range(0, len(y), segment_samples):
            y_segment = y[i:i+segment_samples]
            if len(y_segment) == segment_samples:  # Ensure segment is exactly 3 seconds
                features = extract_features(y_segment, sr)
                features_df = pd.DataFrame([features])

                # Predict genre for the segment and decode it to genre name
                preds_encoded = xgb_loaded.predict(features_df)
                preds_decoded = le_loaded.inverse_transform(preds_encoded)
                genre_counts.append(preds_decoded[0])  # Store predicted genre

        # Count occurrences of each genre and calculate confidence
        genre_frequency = Counter(genre_counts)
        total_segments = len(genre_counts)
        genre_confidence = {genre: round((count / total_segments) * 100, 2) for genre, count in genre_frequency.items()}

        # Display the results
        st.write("Predicted Genre Confidence:")
        st.write(pd.DataFrame.from_dict(genre_confidence, orient='index', columns=['Confidence (%)']))
        
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
    finally:
        # Clean up temporary file
        os.remove(temp_file_path)
