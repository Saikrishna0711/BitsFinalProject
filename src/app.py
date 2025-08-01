#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import torch
import pathlib
import sys

# --- Add project root to path to import other scripts ---
# This assumes 'app.py' is in the 'src' directory. Adjust if needed.
project_root = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root / "src"))

# --- Import refactored functions from your project ---
# NOTE: You would need to refactor your scripts into importable functions.
# Below are placeholder functions to illustrate the concept.
from asr_diarize import run_whisper, run_diarization, merge #
from hf_emotion import classify as classify_emotions #
from transcript2json import convert as transcript2json #

# --- UI Configuration ---
st.set_page_config(page_title="Emotion-Pipe Dashboard", layout="wide")
st.title("🗣️ Emotion-Pipe: Audio Analysis Dashboard")

# Emotion to color mapping for the UI
EMOTION_COLORS = {
    "anger": "red", "disgust": "purple", "fear": "orange",
    "joy": "yellow", "sadness": "blue", "surprise": "green",
    "neutral": "grey", "love": "pink", "others": "grey"
}

def color_emotion(emotion):
    color = EMOTION_COLORS.get(emotion.lower(), "white")
    return f'background-color: {color}'

# --- Main Application Logic ---
# Sidebar for file upload and settings
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload an audio (.wav) or transcript (.srt, .vtt) file", type=["wav", "srt", "vtt"])
    hf_token = st.text_input("Hugging Face Token", type="password", help="Required for diarization on audio files.")
    run_button = st.button("Analyze Now", disabled=not uploaded_file)

# Main panel for results
if run_button:
    if not uploaded_file:
        st.warning("Please upload a file first.")
    else:
        # Create a temporary directory to store the file
        temp_dir = pathlib.Path("temp")
        temp_dir.mkdir(exist_ok=True)
        file_path = temp_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Running analysis pipeline... This may take a few minutes."):
            try:
                # --- Step 1: ASR + Diarization or Transcript Conversion ---
                if file_path.suffix.lower() == ".wav":
                    if not hf_token:
                        st.error("Hugging Face token is required for audio file processing.")
                        st.stop()
                    st.info("Step 1/3: Transcribing audio with Whisper...")
                    txt_segs = run_whisper(str(file_path)) #
                    st.info("Step 2/3: Running speaker diarization with PyAnnote...")
                    dia = run_diarization(str(file_path), hf_token) #
                    segments = merge(txt_segs, dia) #
                else:
                    st.info("Step 1/3: Converting transcript to segments...")
                    # This function needs to return the path to the JSON file it creates
                    json_path = transcript2json(str(file_path)) #
                    import json
                    segments = json.load(open(json_path))

                if not segments:
                    st.error("No segments were found in the source file.")
                    st.stop()

                # --- Step 2: Emotion Inference ---
                st.info("Step 3/3: Classifying emotions...")
                texts = [seg["text"] for seg in segments]
                emotions = classify_emotions(texts, device="cuda" if torch.cuda.is_available() else "cpu") #
                
                for seg, emo in zip(segments, emotions):
                    seg["emotion"] = emo.lower()

                # --- Step 3: Display Results ---
                st.success("✅ Analysis Complete!")
                
                # Create a DataFrame for display
                df = pd.DataFrame(segments)
                df = df[["start", "end", "speaker", "text", "emotion"]]
                df["start"] = df["start"].apply(lambda x: f"{x:.2f}")
                df["end"] = df["end"].apply(lambda x: f"{x:.2f}")

                # Display styled table
                st.subheader("Emotion Transcript")
                st.dataframe(df.style.applymap(color_emotion, subset=['emotion']), use_container_width=True)

                # Display emotion distribution chart
                st.subheader("Emotion Distribution")
                emotion_counts = df['emotion'].value_counts()
                st.bar_chart(emotion_counts)

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")

else:
    st.info("Upload a file and click 'Analyze Now' to see the results.")