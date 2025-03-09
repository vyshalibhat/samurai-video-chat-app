# backend/app/model.py
# (Optional) Could unify or just remove this file if not needed.

import os
import torch
import torch.nn as nn
import torchvision.models.video as models
import librosa
from faster_whisper import WhisperModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class EmotionResNet3DLocal:
    def __init__(self, model_path="6emotions_resnet3dV2.pth"):
        base_dir = os.path.dirname(__file__)
        full_model_path = os.path.join(base_dir, model_path)

class SpeechToTextModel:
    """
    Example for Model 2 if you prefer a separate class.
    Not placeholders: real transcribe() logic must come from your code.
    """
    def __init__(self, stt_path="SpeechTranscription.pth"):
        # Load .pth if needed, or use faster-whisper
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model = WhisperModel("base", device=self.device, compute_type="float16")

    def transcribe(self, wav_path):
        audio, _ = librosa.load(wav_path, sr=16000)
        segments, info = self.whisper_model.transcribe(audio)
        text = " ".join([seg.text for seg in segments])
        return text

class LLMModel:
    """
    Example for Model 3 if you want to define it here.
    Not placeholders: real generate_response() logic must come from your code.
    """
    def __init__(self, llm_path="dementiahelperllm.pth"):
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
        except Exception:
            return "I'm having trouble responding."