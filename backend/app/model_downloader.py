# backend/app/model_downloader.py

import os
import requests
import torch
import torch.nn as nn
import torchvision.models.video as models
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# URLs to your Hugging Face repos / .pth files
# 1) Emotion Model
EMOTION_HF_URL = "https://huggingface.co/games7777777/samurAI_model1/resolve/main/6emotions_resnet3dV2.pth"

# 3) DementiaHelper LLM (3rd model)
LLM_HF_URL = "https://huggingface.co/Joylim/DementiaHelperLLM/resolve/main/dementiahelperllm.pth"

# -------------------------------------------------------------------
# Download Helpers
# -------------------------------------------------------------------
def download_emotion_pth(model_path):
    """Download the .pth file for the emotion model if not found locally."""
    print(f"[INFO] {model_path} not found locally. Downloading from: {EMOTION_HF_URL}")
    r = requests.get(EMOTION_HF_URL, stream=True)
    with open(model_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
    print("[INFO] Emotion .pth download complete.")


def download_llm_pth(model_path):
    """Download the .pth file for the 3rd model (LLM) if not found locally."""
    print(f"[INFO] {model_path} not found locally. Downloading from: {LLM_HF_URL}")
    r = requests.get(LLM_HF_URL, stream=True)
    with open(model_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
    print("[INFO] LLM .pth download complete.")

# -------------------------------------------------------------------
# 1) Emotion Model Class
# -------------------------------------------------------------------
class EmotionResNet3D:
    def __init__(self, model_path="6emotions_resnet3dV2.pth"):
        base_dir = os.path.dirname(__file__)
        full_model_path = os.path.join(base_dir, model_path)

        # If not found locally, fetch from Hugging Face
        if not os.path.exists(full_model_path):
            download_emotion_pth(full_model_path)

        # Load checkpoint
        checkpoint = torch.load(full_model_path, map_location="cpu")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.r3d_18(pretrained=False)

        # Suppose you have 6 classes for emotions
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 6)

        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

        self.emotions = ['angry', 'calm', 'fearful', 'sad', 'happy', 'neutral']

    def predict(self, input_tensor):
        """
        Accepts a 5D tensor: (B, C, T, H, W).
        Returns raw logits (un-softmaxed).
        """
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
        return output

# -------------------------------------------------------------------
# 3) DementiaHelperLLM (3rd Model)
# -------------------------------------------------------------------
class DementiaHelperLLM:
    """
    Example class for your 3rd model. Adjust the generation logic,
    pipeline, or prompt-engineering to fit your exact needs.
    """
    def __init__(self, model_path="dementiahelperllm.pth"):
        base_dir = os.path.dirname(__file__)
        full_model_path = os.path.join(base_dir, model_path)

        # If not found locally, fetch from Hugging Face
        if not os.path.exists(full_model_path):
            download_llm_pth(full_model_path)

        print("[INFO] Loading DementiaHelperLLM checkpoint from local file.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Example: load a Transformers-based model – adapt to your LLM
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        # If you want to load custom .pth weights:
        #   self.model.load_state_dict(torch.load(full_model_path, map_location=self.device))

        self.model.to(self.device)
        self.model.eval()

        self.generation_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )

    def generate_response(self, user_text: str, emotion: str) -> str:
        """
        Generate a response from the LLM using the user’s transcribed text
        and the detected emotion as part of the prompt.
        """
        prompt = (
            f"You are a compassionate assistant. The user is feeling {emotion}.\n"
            f"User says: '{user_text}'\n"
            "Assistant:"
        )

        outputs = self.generation_pipeline(
            prompt,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        response = outputs[0]['generated_text']
        response = response.split("Assistant:")[-1].strip()
        return response
