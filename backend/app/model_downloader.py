# backend/app/model_downloader.py

import os
import requests
import torch
import torch.nn as nn
import torchvision.models.video as models
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# URLs to your Hugging Face repos / .pth files
EMOTION_HF_URL = "https://huggingface.co/games7777777/samurAI_model1/resolve/main/6emotions_resnet3dV2.pth"
LLM_HF_URL = "https://huggingface.co/Joylim/DementiaHelperLLM/resolve/main/dementiahelperllm7.pth"


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


###############################################################################
# 1) Emotion Model Class
###############################################################################
class EmotionResNet3D:
    def __init__(self, model_path="6emotions_resnet3dV2.pth"):
        base_dir = os.path.dirname(__file__)
        full_model_path = os.path.join(base_dir, model_path)

        print(f"[INFO] Checking for emotion model at: {full_model_path}")
        if not os.path.exists(full_model_path):
            download_emotion_pth(full_model_path)
        else:
            print(f"[INFO] Found local emotion model: {full_model_path}")

        checkpoint = torch.load(full_model_path, map_location="cpu")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.r3d_18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 6)

        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

        # Match these labels with however you trained your model
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


###############################################################################
# 2) DementiaHelper LLM (Caretaker-Style, short + passionate)
###############################################################################
class DementiaHelperLLM:
    """
    A caretaker-style LLM class that returns a short, empathetic response.
    If you want your custom .pth weights to override GPT2, uncomment load_state_dict.
    """
    def __init__(self, model_path="dementiahelperllm7.pth"):
        base_dir = os.path.dirname(__file__)
        full_model_path = os.path.join(base_dir, model_path)

        print(f"[INFO] Checking for LLM model at: {full_model_path}")
        if not os.path.exists(full_model_path):
            download_llm_pth(full_model_path)
        else:
            print(f"[INFO] Found local LLM file: {full_model_path}")

        print("[INFO] Loading DementiaHelperLLM checkpoint from local file.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Base architecture
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        # If you want to load the actual .pth weights:
        #   self.model.load_state_dict(torch.load(full_model_path, map_location=self.device))

        self.model.to(self.device)
        self.model.eval()

        # Create a text-generation pipeline
        self.generation_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )

    def generate_response(self, user_text: str, emotion: str) -> str:
        """
        Generate a short, passionate, empathetic caretaker response.
        If the user is sad/tired, mention rest or encouragement in 1-2 lines.
        Optionally mention a real link (sleepfoundation.org or verywellmind.com).
        Keep the temperature low so it's short and direct.
        """
        prompt = (
            "You are a warm, compassionate caregiver talking to an adult with dementia. "
            "They feel {emotion}, mention being tired or overwhelmed. "
            "Give a short, uplifting message (1-2 lines) that shows empathy. "
            "Suggest rest or reassurance. If needed, mention a real link like https://www.sleepfoundation.org. "
            "No rhetorical questions or tangents.\n\n"
            f"User's emotion: {emotion}\n"
            f"User says:\n{user_text}\n\n"
            "Assistant:"
        ).format(emotion=emotion)

        outputs = self.generation_pipeline(
            prompt,
            max_new_tokens=40,      # short output
            do_sample=True,
            temperature=0.1,        # keeps it direct
            top_p=0.5,
            top_k=20,
            repetition_penalty=1.4,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

        response = outputs[0]['generated_text']

        # Remove everything before/including "Assistant:"
        if "Assistant:" in response:
            response = response.split("Assistant:", 1)[-1].strip()

        return response
