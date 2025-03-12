# upload_to_huggingface.py

from huggingface_hub import HfApi

def upload_emotion_model():
    """Upload the 1st model (emotion) to HF."""
    api = HfApi()
    api.upload_file(
        path_or_fileobj="6emotions_resnet3dV2.pth",
        path_in_repo="6emotions_resnet3dV2.pth",
        repo_id="games7777777/samurAI_model1",  # Your 1st model's repo
        repo_type="model",
    )
    print("Emotion model upload complete!")

ddef upload_llm_model():
    """Upload the 3rd model (LLM) to HF."""
    api = HfApi()
    api.upload_file(
        path_or_fileobj="dementiahelperllm7.pth",  # local filename
        path_in_repo="dementiahelperllm7.pth",     # how it appears on HF
        repo_id="Joylim/DementiaHelperLLM",        # Your 3rd model's repo
        repo_type="model",
    )
    print("LLM model upload complete!")

def main():
    # Decide which model(s) you want to upload
    upload_emotion_model()
    # upload_llm_model()

if __name__ == "__main__":
    main()
