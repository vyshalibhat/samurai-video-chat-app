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
