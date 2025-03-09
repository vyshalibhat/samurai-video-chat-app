# backend/upload_to_huggingface_draft.py

from huggingface_hub import HfApi

def main():
    """
    Upload the emotion model .pth to Hugging Face
    (same logic as your original code for model1).
    """
    api = HfApi()
    api.upload_file(
        path_or_fileobj="6emotions_resnet3dV2.pth",
        path_in_repo="6emotions_resnet3dV2.pth",
        repo_id="games7777777/samurAI_model1",  # your repo ID for Model 1
        repo_type="model",
    )
    print("Upload complete for 6emotions_resnet3dV2.pth!")

def upload_speech_to_text_model():
    """
    Upload the SpeechTranscription.pth model to Hugging Face.
    """
    api = HfApi()
    api.upload_file(
        path_or_fileobj="SpeechTranscription.pth",
        path_in_repo="SpeechTranscription.pth",
        repo_id="Bhavna1998/WhisperX",  # your repo ID for STT
        repo_type="model",
    )
    print("Speech-to-Text model upload complete!")

def upload_llm_model():
    """
    Upload the dementiahelperllm.pth model to Hugging Face.
    """
    api = HfApi()
    api.upload_file(
        path_or_fileobj="dementiahelperllm.pth",
        path_in_repo="dementiahelperllm.pth",
        repo_id="Joylim/DementiaHelperLLM",  # your repo ID for LLM
        repo_type="model",
    )
    print("LLM model upload complete!")

if __name__ == "__main__":
    # Now we invoke all three
    main()                         # Upload emotion model
    upload_speech_to_text_model()  # Upload STT model
    upload_llm_model()            # Upload LLM model
