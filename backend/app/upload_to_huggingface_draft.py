# upload_to_huggingface.py

from huggingface_hub import HfApi

def main():
    api = HfApi()
    api.upload_file(
    path_or_fileobj="6emotions_resnet3dV2.pth",
    path_in_repo="6emotions_resnet3dV2.pth",
    repo_id="games7777777/samurAI_model1",  # <-- your repo ID
    repo_type="model",
    )
    print("Upload complete!")

if __name__ == "__main__":
    main()

def upload_speech_to_text_model():
    """
    Upload the SpeechTranscription.pth model to Hugging Face.
    Adjust the path_in_repo or repo_id as needed.
    """
    api = HfApi()
    api.upload_file(
        path_or_fileobj="SpeechTranscription.pth",
        path_in_repo="SpeechTranscription.pth",
        repo_id="Bhavna1998/WhisperX",  # <-- your repo ID
        repo_type="model"
    )
    print("Speech-to-Text model upload complete!")

def upload_llm_model():
    """
    Upload the dementiahelperllm.pth model to Hugging Face.
    Adjust the path_in_repo or repo_id as needed.
    """
    api = HfApi()
    api.upload_file(
        path_or_fileobj="dementiahelperllm.pth",
        path_in_repo="dementiahelperllm.pth",
        repo_id="Joylim/DementiaHelperLLM",  # <-- your repo ID
        repo_type="model"
    )
    print("LLM model upload complete!")
