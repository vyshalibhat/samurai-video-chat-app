from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import ffmpeg
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from .model import EmotionResNet3D  # from your model.py

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

emotion_model = EmotionResNet3D(model_path="6emotions_resnet3dV2.pth")

def convert_to_mp4(input_path: str, output_path: str):
    """
    Use ffmpeg-python to convert the input file (e.g. .webm) 
    to .mp4 with H.264 (libx264) and AAC audio.
    """
    stream = ffmpeg.input(input_path)
    # vcodec='libx264' for H.264, acodec='aac' for AAC
    stream = ffmpeg.output(stream, output_path, vcodec='libx264', acodec='aac', strict='-2')
    ffmpeg.run(stream, overwrite_output=True)
    return output_path

def process_video(video_path: str):
    """
    This function now expects video_path to be an mp4 file 
    with a known, decodable codec (like H.264).
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("OpenCV could NOT open the file at all:", video_path)
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames read by OpenCV:", total_frames)

    # We'll sample 8 frames
    indices = np.linspace(0, total_frames - 1, 8).astype(int)
    frames = []
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id in indices:
            # BGR -> RGB, then resize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (112, 112))
            frames.append(frame)
        frame_id += 1
    cap.release()

    if len(frames) < 8:
        return None  # not enough frames

    frames = np.array(frames, dtype=np.float32) / 255.0
    frames = np.transpose(frames, (3, 0, 1, 2))
    input_tensor = torch.tensor(frames).unsqueeze(0)
    return input_tensor

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    """
    1. We first save the incoming .webm or .mp4 
    2. Convert it to a standard .mp4 (H.264) with FFmpeg
    3. Then run process_video() with OpenCV
    """
    try:
        # Step A: Save the uploaded file to a temp path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_video:
            temp_video.write(await file.read())
            input_path = temp_video.name

        # Step B: Convert that file to .mp4
        output_path = input_path.replace(".webm", ".mp4")
        convert_to_mp4(input_path, output_path)
        # Clean up the original .webm file
        os.remove(input_path)

        # Step C: Process the new .mp4 with OpenCV
        input_tensor = process_video(output_path)
        os.remove(output_path)  # optionally delete the .mp4 after processing

        if input_tensor is None:
            return {"error": "Insufficient frames in video or decode failure."}

        # Step D: Inference
        with torch.no_grad():
            logits = emotion_model.predict(input_tensor)
            probs = F.softmax(logits[0], dim=0)

            # Same order of classes as in model.py
            emotions = emotion_model.emotions
            scores = {emotions[i]: float(probs[i]) for i in range(len(emotions))}
            predicted_emotion = max(scores, key=scores.get)

            # Debug line: Print each emotion and probability ## COMMENT LATER AS THIS ONE FOR DEBUGGING PURPOSES
            print("Emotion Probabilities (Debug):")
            for emotion, score in scores.items():
                print(f"  {emotion}: {score:.4f}")

        return {
            "predicted_emotion": predicted_emotion,
            "scores": scores
        }
    except Exception as e:
        return {"error": str(e)}
