# backend/app/main.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

# Here we would import the model + Google Drive logic
# If you prefer, you can import from "model.py" or unify them


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


@app.get("/")
def read_root():
    return {"message": "Backend API is running"}


@app.options("/predict")
async def options_predict():
    # Handle preflight requests
    return {}


# Instantiate your model (this triggers the .pth check / GDrive download if needed)
try:
    emotion_model = EmotionResNet3D(model_path="6emotions_resnet3dV2.pth")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    # Fallback to a mock model for testing
    emotion_model = EmotionResNet3D(model_path="mock_path")


def convert_to_mp4(input_path: str, output_path: str):
    """
    Use ffmpeg-python to convert the input file (e.g. .webm) 
    to .mp4 (H.264 + AAC).
    """
    try:
        # Ensure input file exists
        if not os.path.exists(input_path):
            print(f"Input file does not exist: {input_path}")
            return None

        # Make sure the conversion happens with proper shell escape
        import subprocess
        cmd = [
            "ffmpeg", "-i", input_path, "-c:v", "libx264", "-c:a", "aac",
            "-strict", "-2", output_path, "-y"
        ]
        print(f"Running command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd,
                                    capture_output=True,
                                    text=True,
                                    check=True)
            print(f"FFMPEG stdout: {result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"FFMPEG error: {e}")
            print(f"FFMPEG stderr: {e.stderr}")
            return None

        # Verify the output was created
        if os.path.exists(output_path):
            print(f"Successfully converted to: {output_path}")
            return output_path
        else:
            print(f"Failed to create output file: {output_path}")
            return None
    except Exception as e:
        print(f"Error in convert_to_mp4: {str(e)}")
        return None


def process_video(video_path: str):
    """
    We'll read the .mp4 file with OpenCV, sample 8 frames, and build a tensor.
    """
    try:
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
    except Exception as e:
        print(f"Error in process_video: {str(e)}")
        return None


@app.get("/")
def read_root():
    return {"message": "Emotion Analysis API"}

# Fix: pass `model_size_or_path` as the first param
whisper_model = WhisperModel(
    model_size_or_path="base",   # or 'tiny', 'small', 'medium', 'large'
    device=device,
    compute_type=compute_type
)

def extract_audio_from_video(video_path: str, audio_path: str):
    command = f'ffmpeg -i "{video_path}" -q:a 0 -map a "{audio_path}" -y'
    subprocess.run(command, shell=True, check=True)


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


@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    """
    1) Save the uploaded file to a temp .webm
    2) Convert to mp4 w/ H.264
    3) Process frames with OpenCV
    4) Run inference
    """
    try:
        print(
            f"Received file: {file.filename}, content_type: {file.content_type}"
        )

        # Step A: Save the incoming file
        with tempfile.NamedTemporaryFile(delete=False,
                                         suffix=".webm") as temp_video:
            content = await file.read()
            print(f"File size: {len(content)} bytes")
            temp_video.write(content)
            input_path = temp_video.name

        print(f"Saved to: {input_path}")

        # Step B: Convert to mp4
        output_path = input_path.replace(".webm", ".mp4")
        conversion_result = convert_to_mp4(input_path, output_path)

        if not conversion_result:
            return {"error": "Failed to convert video format"}

        # Keep the original file until we're sure conversion worked
        try:
            os.remove(input_path)  # remove the original .webm
            print(f"Removed original file: {input_path}")
        except Exception as e:
            print(f"Could not remove original file: {e}")

        # Step C: Prepare frames
        input_tensor = process_video(output_path)

        try:
            os.remove(output_path)  # optional cleanup
            print(f"Removed converted file: {output_path}")
        except Exception as e:
            print(f"Could not remove converted file: {e}")

        if input_tensor is None:
            return {"error": "Insufficient frames or decode failure."}

        if input_tensor is None:
            return {"error": "Insufficient frames or decode failure."}

        # Step D: Inference
        with torch.no_grad():
            logits = emotion_model.predict(input_tensor)
            probs = F.softmax(logits[0], dim=0)

            emotions = emotion_model.emotions
            scores = {
                emotions[i]: float(probs[i])
                for i in range(len(emotions))
            }
            predicted_emotion = max(scores, key=scores.get)

            # Debug print
            print("Emotion Probabilities:")
            for emotion, score_val in scores.items():
                print(f"  {emotion}: {score_val:.4f}")

        return {"predicted_emotion": predicted_emotion, "scores": scores}

    except Exception as e:
        print(f"Error in predict_emotion: {str(e)}")
        return {"error": str(e)}


@app.post("/process_all")
async def process_all(file: UploadFile = File(...)):
    """
    1) Emotion (model 1)
    2) Transcribe (faster-whisper, model 2)
    3) LLM (model 3)
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False,
                                         suffix=".webm") as tmp_vid:
            tmp_vid.write(await file.read())
            webm_path = tmp_vid.name

        mp4_path = webm_path.replace(".webm", ".mp4")
        convert_to_mp4(webm_path, mp4_path)
        os.remove(webm_path)

        # EMOTION
        input_tensor = process_video_for_emotion(mp4_path)
        if input_tensor is None:
            os.remove(mp4_path)
            raise HTTPException(
                status_code=500,
                detail=
                "Insufficient frames or decode failure for emotion model.")

        with torch.no_grad():
            logits = emotion_model.predict(input_tensor)
            probs = F.softmax(logits[0], dim=0)
            emotions = emotion_model.emotions
            scores = {
                emotions[i]: float(probs[i])
                for i in range(len(emotions))
            }
            predicted_emotion = max(scores, key=scores.get)

        # STT
        with tempfile.NamedTemporaryFile(delete=False,
                                         suffix=".wav") as tmp_wav:
            wav_path = tmp_wav.name

        extract_audio_from_video(mp4_path, wav_path)
        os.remove(mp4_path)

        segments, info = whisper_model.transcribe(wav_path)
        os.remove(wav_path)
        transcription = " ".join(seg.text for seg in segments)

        # LLM
        llm_response = llm_model.generate_response(transcription,
                                                   predicted_emotion)

        return {
            "predicted_emotion": predicted_emotion,
            "transcription": transcription,
            "llm_response": llm_response
        }

    except subprocess.CalledProcessError as ffmpeg_err:
        raise HTTPException(
            status_code=500,
            detail=f"ffmpeg failed to extract audio: {str(ffmpeg_err)}")
    except Exception as e:
        print("ERROR in /process_all:", repr(e))
        raise HTTPException(status_code=500,
                            detail=f"Error during processing: {str(e)}")


@app.post("/transcribe")
async def transcribe_video(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False,
                                         suffix=".webm") as tmp_vid:
            tmp_vid.write(await file.read())
            webm_path = tmp_vid.name

        mp4_path = webm_path.replace(".webm", ".mp4")
        convert_to_mp4(webm_path, mp4_path)
        os.remove(webm_path)

        with tempfile.NamedTemporaryFile(delete=False,
                                         suffix=".wav") as tmp_wav:
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
            detail=f"ffmpeg failed to extract audio: {str(ffmpeg_err)}")
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Error during transcription: {str(e)}")
