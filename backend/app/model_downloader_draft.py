# model_downloader.py

import os
import requests
import torch
import torch.nn as nn
import torchvision.models.video as models

# URL to your .pth on Hugging Face
# e.g. https://huggingface.co/games7777777/samurAI_model1/resolve/main/6emotions_resnet3dV2.pth
HF_URL = "https://huggingface.co/games7777777/samurAI_model1/resolve/main/6emotions_resnet3dV2.pth"

def download_from_huggingface(model_path):
    """
    Download the .pth file from Hugging Face if not found locally.
    """
    print(f"[INFO] {model_path} not found locally. Downloading from Hugging Face...")
    r = requests.get(HF_URL, stream=True)
    with open(model_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
    print("[INFO] Download complete.")

class EmotionResNet3D:
    def __init__(self, model_path="6emotions_resnet3dV2.pth"):
        base_dir = os.path.dirname(__file__)
        full_model_path = os.path.join(base_dir, model_path)

        # If not found locally, fetch from Hugging Face
        if not os.path.exists(full_model_path):
            download_from_huggingface(full_model_path)

        # If you're on PyTorch 2.6+ & trust your file => weights_only=False
        checkpoint = torch.load(full_model_path, map_location="cpu", weights_only=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.r3d_18(pretrained=False)

        # Suppose you have 6 classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 6)

        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

        self.emotions = ['angry', 'calm', 'fearful', 'sad', 'happy', 'neutral']

    def predict(self, input_tensor):
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
        return output

HF2_URL = "https://huggingface.co/Bhavna1998/WhisperX/resolve/main/SpeechTranscription.pth"

def download_stt_model(local_path="SpeechTranscription.pth"):
    """
    Checks if the STT .pth file exists locally; if not, downloads from HF2_URL.
    """
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"Downloading SpeechTranscription.pth to {local_path}...")
        r = requests.get(HF2_URL)
        with open(local_path, "wb") as f:
            f.write(r.content)
        print("Download complete.")
    else:
        print("SpeechTranscription.pth already present.")

class WhisperXModel:
    """
    Example class for your Speech-to-Text model loading and inference.
    Adjust to match your actual code that uses 'SpeechTranscription.pth'.
    """
    def __init__(self, model_path="SpeechTranscription.pth"):
        # e.g., load the local .pth if you have a custom model, 
        # or use standard 'faster-whisper' if it's just a pipeline.
        # This is a placeholder; adapt as needed:
        print(f"Loading STT model from {model_path}...")
        # Real code might be: self.model = MySTTClass(...)
        # state_dict = torch.load(model_path, map_location="cpu")
        # self.model.load_state_dict(state_dict)
        # self.model.eval()

    def transcribe(self, wav_path):
        """
        Given a .wav file, return the transcribed text. 
        """
        # Real code for STT inference goes here
        return "<transcribed-text-placeholder>"
    
HF3_URL = "https://huggingface.co/Joylim/DementiaHelperLLM/resolve/main/dementiahelperllm.pth"

class DementiaHelperLLM:
    """
    Example class for your LLM. 
    Adjust to match your actual code that uses 'dementiahelperllm.pth'.
    """
    def __init__(self, model_path="dementiahelperllm.pth"):
        print(f"Loading Gemma LLM from {model_path}...")
        # state_dict = torch.load(model_path, map_location="cpu")
        # self.load_state_dict(state_dict)
        # self.eval()

    def generate_response(self, emotion, user_text):
        """
        Combine the user's emotion + spoken text to produce a custom reply.
        """
        # Real code to produce an LLM-based response
        return f"<LLM response based on emotion={emotion}, text={user_text}>"
    
#LLM

 

