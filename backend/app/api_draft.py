# backend/app/api_draft.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import subprocess

# Import the same variables/functions for the emotion model, STT, and LLM from model_downloader_draft
# We DO NOT rename or alter the emotion model's variables or logic.
from .model_downloader_draft import (
    emotion_model,           # This is the existing emotion model (Model 1) - unchanged
    speech_to_text_model,    # Unified approach for STT (Model 2)
    llm_model,               # LLM (Model 3)
    convert_webm_to_mp4,     # Function to unify .webm -> .mp4
    extract_audio_from_video # Function to extract WAV from MP4
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def process_video_frames(mp4_path: str, clip_length: int = 8):
    """
    Reads an .mp4 file with OpenCV, samples 'clip_length' frames, 
    and returns a tensor for the emotion model.
    We DO NOT rename or change any logic of the emotion model itself.
    """
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open file: {mp4_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise ValueError("No frames found in video.")

    # Sample frames uniformly
    indices = np.linspace(0, total_frames - 1, clip_length).astype(int)
    frames = []
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id in indices:
            # BGR -> RGB, then resize to match the emotion model's expected size
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (112, 112))
            frames.append(frame)
        frame_id += 1
    cap.release()

    if len(frames) < clip_length:
        raise ValueError("Insufficient frames for emotion detection.")

    frames_np = np.array(frames, dtype=np.float32) / 255.0
    # (clip_length, 112, 112, 3) -> (3, clip_length, 112, 112)
    frames_np = np.transpose(frames_np, (3, 0, 1, 2))
    input_tensor = torch.tensor(frames_np).unsqueeze(0)  # (1, 3, T, H, W)
    return input_tensor

@app.post("/predict")
async def predict_emotion_stt_llm(file: UploadFile = File(...)):
    """
    Single endpoint that:
    1) Accepts .webm or .mp4
    2) If .webm, converts to .mp4
    3) Runs emotion detection (Model 1) 
    4) Extracts audio -> speech-to-text (Model 2)
    5) Passes emotion + transcript -> LLM (Model 3)
    6) Returns JSON with emotion, STT transcript, and LLM response.
    """
    # Check content type
    if file.content_type not in ["video/webm", "video/mp4"]:
        raise HTTPException(status_code=400, detail="Upload .webm or .mp4 only.")

    temp_path = None
    mp4_path = None
    wav_path = None

    try:
        # Save uploaded file
        suffix = ".webm" if file.content_type == "video/webm" else ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_vid:
            temp_vid.write(await file.read())
            temp_path = temp_vid.name

        # Convert if it's .webm
        if suffix == ".webm":
            mp4_path = temp_path.replace(".webm", "_converted.mp4")
            convert_webm_to_mp4(temp_path, mp4_path)
        else:
            mp4_path = temp_path  # already mp4

        # Emotion Detection (Model 1)
        input_tensor = process_video_frames(mp4_path)
        logits = emotion_model.predict(input_tensor)
        probs = F.softmax(logits[0], dim=0)
        emotions = emotion_model.emotions
        scores = {emotions[i]: float(probs[i]) for i in range(len(emotions))}
        predicted_emotion = max(scores, key=scores.get)

        # Speech-to-Text (Model 2)
        wav_path = mp4_path.replace(".mp4", ".wav")
        extract_audio_from_video(mp4_path, wav_path)
        transcript = speech_to_text_model.transcribe(wav_path)

        # LLM (Model 3)
        llm_response = llm_model.generate_response(predicted_emotion, transcript)

        return JSONResponse({
            "predicted_emotion": predicted_emotion,
            "emotion_scores": scores,
            "transcribed_text": transcript,
            "llm_response": llm_response
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        if mp4_path and os.path.exists(mp4_path) and mp4_path != temp_path:
            os.remove(mp4_path)
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
