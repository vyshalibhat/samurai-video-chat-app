# backend/app/model_downloader_draft.py

import os
import requests
import torch
import torch.nn as nn
import torchvision.models.video as models
import librosa
import numpy as np
import subprocess
from faster_whisper import WhisperModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

###########################################
# EMOTION MODEL (UNCHANGED)
###########################################
# We do NOT rename or modify any part of the emotion model logic.
EMOTION_MODEL_URL = "https://huggingface.co/YourUser/YourRepo/resolve/main/6emotions_resnet3dV2.pth"

class EmotionResNet3D:
    def __init__(self, model_path="6emotions_resnet3dV2.pth"):
        base_dir = os.path.dirname(__file__)
        self.full_model_path = os.path.join(base_dir, model_path)
        if not os.path.exists(self.full_model_path):
            self.download_emotion_model()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.r3d_18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 6)
        checkpoint = torch.load(self.full_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

        self.emotions = ['angry', 'calm', 'fearful', 'sad', 'happy', 'neutral']

    def download_emotion_model(self):
        print(f"[INFO] Downloading emotion model from {EMOTION_MODEL_URL} ...")
        r = requests.get(EMOTION_MODEL_URL, stream=True)
        with open(self.full_model_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("[INFO] Emotion model download complete.")

    def predict(self, input_tensor):
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            outputs = self.model(input_tensor)
        return outputs

###########################################
# STT MODEL
###########################################
STT_MODEL_URL = "https://huggingface.co/Bhavna1998/WhisperX/resolve/main/SpeechTranscription.pth"

def download_stt_model(local_path="SpeechTranscription.pth"):
    if not os.path.exists(local_path):
        print(f"[INFO] Downloading STT model from {STT_MODEL_URL} to {local_path} ...")
        r = requests.get(STT_MODEL_URL, stream=True)
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("[INFO] STT model download complete.")
    else:
        print("[INFO] STT model already exists locally.")

class WhisperXModel:
    def __init__(self, model_path="SpeechTranscription.pth"):
        base_dir = os.path.dirname(__file__)
        self.full_model_path = os.path.join(base_dir, model_path)
        download_stt_model(self.full_model_path)  # Now we actually invoke it

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        print(f"[INFO] Initializing WhisperXModel with device={self.device}, compute_type={self.compute_type}")
        self.whisper_model = WhisperModel("base", device=self.device, compute_type=self.compute_type)

    def transcribe(self, wav_path: str):
        audio, _ = librosa.load(wav_path, sr=16000)
        segments, info = self.whisper_model.transcribe(audio)
        text = " ".join([seg.text for seg in segments])
        return text

###########################################
# LLM MODEL
###########################################
LLM_MODEL_URL = "https://huggingface.co/Joylim/DementiaHelperLLM/resolve/main/dementiahelperllm.pth"

def download_llm_model(local_path="dementiahelperllm.pth"):
    if not os.path.exists(local_path):
        print(f"[INFO] Downloading LLM model from {LLM_MODEL_URL} to {local_path} ...")
        r = requests.get(LLM_MODEL_URL, stream=True)
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("[INFO] LLM model download complete.")
    else:
        print("[INFO] LLM model already present locally.")

class DementiaHelperLLM:
    def __init__(self, model_path="dementiahelperllm.pth"):
        base_dir = os.path.dirname(__file__)
        self.full_model_path = os.path.join(base_dir, model_path)
        download_llm_model(self.full_model_path)  # Actually invoked

        print("[INFO] Initializing DementiaHelperLLM from huggingface pipeline.")
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
        self.model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map="auto")
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def generate_response(self, emotion, user_text):
        prompt = (
            f"You are a compassionate assistant. The user feels {emotion}.\n"
            f"The user says: {user_text}\nAssistant:"
        )
        try:
            output = self.pipe(prompt, max_new_tokens=300, do_sample=True, temperature=0.7, top_p=0.9)
            return output[0]['generated_text'].split("Assistant:")[-1].strip()
        except Exception as e:
            print(f"[ERROR] LLM error: {e}")
            return "I'm having trouble responding right now."

###########################################
# HELPER FUNCTIONS (for .webm -> .mp4, etc.)
###########################################
def convert_webm_to_mp4(input_path: str, output_path: str):
    cmd = f'ffmpeg -i "{input_path}" -c:v libx264 -c:a aac "{output_path}" -y'
    subprocess.run(cmd, shell=True, check=True)

def extract_audio_from_video(video_path: str, wav_path: str):
    cmd = f'ffmpeg -i "{video_path}" -q:a 0 -map a "{wav_path}" -y'
    subprocess.run(cmd, shell=True, check=True)

###########################################
# INSTANTIATE ALL MODELS (No placeholders)
###########################################
emotion_model = EmotionResNet3D()      # Model 1, unchanged
speech_to_text_model = WhisperXModel() # Model 2
llm_model = DementiaHelperLLM()        # Model 3
