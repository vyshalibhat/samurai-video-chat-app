# backend/app/api.py

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import ffmpeg
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Here we import the model + Google Drive logic
# If you prefer, you can import from "model.py" or unify them
from .model_downloader import EmotionResNet3D

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate your model (this triggers the .pth check / GDrive download if needed)
emotion_model = EmotionResNet3D(model_path="6emotions_resnet3dV2.pth")


def convert_to_mp4(input_path: str, output_path: str):
    """
    Use ffmpeg-python to convert the input file (e.g. .webm) 
    to .mp4 (H.264 + AAC).
    """
    stream = ffmpeg.input(input_path)
    stream = ffmpeg.output(stream, output_path, vcodec='libx264', acodec='aac', strict='-2')
    ffmpeg.run(stream, overwrite_output=True)
    return output_path


def process_video(video_path: str):
    """
    We'll read the .mp4 file with OpenCV, sample 8 frames, and build a tensor.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("OpenCV could NOT open the file:", video_path)
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames read by OpenCV:", total_frames)

    # We'll sample 8 frames across the entire clip
    indices = np.linspace(0, total_frames - 1, 8).astype(int)
    frames = []
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id in indices:
            # Convert BGR -> RGB, then resize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (112, 112))
            frames.append(frame)
        frame_id += 1
    cap.release()

    # If fewer than 8 frames were read, we can't proceed
    if len(frames) < 8:
        return None

    frames = np.array(frames, dtype=np.float32) / 255.0
    # (8, 112, 112, 3) -> (3, 8, 112, 112)
    frames = np.transpose(frames, (3, 0, 1, 2))
    # add a batch dimension -> (1, 3, 8, 112, 112)
    input_tensor = torch.tensor(frames).unsqueeze(0)
    return input_tensor


@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    """
    1) Save the uploaded file to a temp .webm
    2) Convert to mp4 w/ H.264
    3) Process frames with OpenCV
    4) Run inference
    """
    try:
        # Step A: Save the incoming file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_video:
            temp_video.write(await file.read())
            input_path = temp_video.name

        # Step B: Convert to mp4
        output_path = input_path.replace(".webm", ".mp4")
        convert_to_mp4(input_path, output_path)
        os.remove(input_path)  # remove the original .webm

        # Step C: Prepare frames
        input_tensor = process_video(output_path)
        os.remove(output_path)  # optional cleanup

        if input_tensor is None:
            return {"error": "Insufficient frames or decode failure."}

        # Step D: Inference
        with torch.no_grad():
            logits = emotion_model.predict(input_tensor)
            probs = F.softmax(logits[0], dim=0)

            emotions = emotion_model.emotions
            scores = {emotions[i]: float(probs[i]) for i in range(len(emotions))}
            predicted_emotion = max(scores, key=scores.get)

            # Debug print
            print("Emotion Probabilities:")
            for emotion, score_val in scores.items():
                print(f"  {emotion}: {score_val:.4f}")

        return {
            "predicted_emotion": predicted_emotion,
            "scores": scores
        }

    except Exception as e:
        return {"error": str(e)}
