# backend/app/api.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import ffmpeg
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import subprocess

# 1) Import the 1st model (Emotion) + 3rd model (LLM)
from .model_downloader import EmotionResNet3D, DementiaHelperLLM

# 2) Import faster-whisper for STT
from faster_whisper import WhisperModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate Emotion model + LLM
emotion_model = EmotionResNet3D(model_path="6emotions_resnet3dV2.pth")
llm_model = DementiaHelperLLM(model_path="dementiahelperllm.pth")

# Choose GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

print("Device set to:", device, "compute_type:", compute_type)

# Fix: pass `model_size_or_path` as the first param
whisper_model = WhisperModel(
    model_size_or_path="base",   # or 'tiny', 'small', 'medium', 'large'
    device=device,
    compute_type=compute_type
)

def convert_to_mp4(input_path: str, output_path: str):
    stream = ffmpeg.input(input_path)
    stream = ffmpeg.output(stream, output_path, vcodec='libx264', acodec='aac', strict='-2')
    ffmpeg.run(stream, overwrite_output=True)
    return output_path

def process_video_for_emotion(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("OpenCV could NOT open the file:", video_path)
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames read by OpenCV:", total_frames)

    indices = np.linspace(0, total_frames - 1, 8).astype(int)
    frames = []
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (112, 112))
            frames.append(frame)
        frame_id += 1
    cap.release()

    if len(frames) < 8:
        return None

    frames = np.array(frames, dtype=np.float32) / 255.0
    frames = np.transpose(frames, (3, 0, 1, 2))  # (3, 8, 112, 112)
    input_tensor = torch.tensor(frames).unsqueeze(0)
    return input_tensor

def extract_audio_from_video(video_path: str, audio_path: str):
    command = f'ffmpeg -i "{video_path}" -q:a 0 -map a "{audio_path}" -y'
    subprocess.run(command, shell=True, check=True)

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_vid:
            tmp_vid.write(await file.read())
            webm_path = tmp_vid.name

        mp4_path = webm_path.replace(".webm", ".mp4")
        convert_to_mp4(webm_path, mp4_path)
        os.remove(webm_path)

        input_tensor = process_video_for_emotion(mp4_path)
        os.remove(mp4_path)

        if input_tensor is None:
            return {"error": "Insufficient frames or decode failure."}

        with torch.no_grad():
            logits = emotion_model.predict(input_tensor)
            probs = F.softmax(logits[0], dim=0)
            emotions = emotion_model.emotions
            scores = {emotions[i]: float(probs[i]) for i in range(len(emotions))}
            predicted_emotion = max(scores, key=scores.get)

        return {
            "predicted_emotion": predicted_emotion,
            "scores": scores
        }

    except Exception as e:
        return {"error": str(e)}

@app.post("/transcribe")
async def transcribe_video(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_vid:
            tmp_vid.write(await file.read())
            webm_path = tmp_vid.name

        mp4_path = webm_path.replace(".webm", ".mp4")
        convert_to_mp4(webm_path, mp4_path)
        os.remove(webm_path)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            wav_path = tmp_wav.name

        extract_audio_from_video(mp4_path, wav_path)
        os.remove(mp4_path)

        segments, info = whisper_model.transcribe(wav_path)
        os.remove(wav_path)

        transcription = " ".join(seg.text for seg in segments)
        return JSONResponse(content={"transcription": transcription})

    except subprocess.CalledProcessError as ffmpeg_err:
        raise HTTPException(
            status_code=500,
            detail=f"ffmpeg failed to extract audio: {str(ffmpeg_err)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during transcription: {str(e)}"
        )

@app.post("/process_all")
async def process_all(file: UploadFile = File(...)):
    """
    1) Emotion (model 1)
    2) Transcribe (faster-whisper, model 2)
    3) LLM (model 3)
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_vid:
            tmp_vid.write(await file.read())
            webm_path = tmp_vid.name

        mp4_path = webm_path.replace(".webm", ".mp4")
        convert_to_mp4(webm_path, mp4_path)
        os.remove(webm_path)

        # EMOTION
        input_tensor = process_video_for_emotion(mp4_path)
        if input_tensor is None:
            os.remove(mp4_path)
            raise HTTPException(status_code=500, detail="Insufficient frames or decode failure for emotion model.")

        with torch.no_grad():
            logits = emotion_model.predict(input_tensor)
            probs = F.softmax(logits[0], dim=0)
            emotions = emotion_model.emotions
            scores = {emotions[i]: float(probs[i]) for i in range(len(emotions))}
            predicted_emotion = max(scores, key=scores.get)

        # STT
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            wav_path = tmp_wav.name

        extract_audio_from_video(mp4_path, wav_path)
        os.remove(mp4_path)

        segments, info = whisper_model.transcribe(wav_path)
        os.remove(wav_path)
        transcription = " ".join(seg.text for seg in segments)

        # LLM
        llm_response = llm_model.generate_response(transcription, predicted_emotion)

        return {
            "predicted_emotion": predicted_emotion,
            "transcription": transcription,
            "llm_response": llm_response
        }

    except subprocess.CalledProcessError as ffmpeg_err:
        raise HTTPException(
            status_code=500,
            detail=f"ffmpeg failed to extract audio: {str(ffmpeg_err)}"
        )
    except Exception as e:
        print("ERROR in /process_all:", repr(e))
        raise HTTPException(
            status_code=500,
            detail=f"Error during processing: {str(e)}"
        )
