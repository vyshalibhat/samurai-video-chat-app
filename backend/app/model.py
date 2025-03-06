# model.py
import os
import torch
import torch.nn as nn
import torchvision.models.video as models

class EmotionResNet3D:
    def __init__(self, model_path="6emotions_resnet3dV2.pth"):
        # Compute absolute path to the .pth file relative to the model.py
        base_dir = os.path.dirname(__file__)
        full_model_path = os.path.join(base_dir, model_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = models.r3d_18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 6)

        checkpoint = torch.load(full_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

        self.emotions = ['angry', 'calm', 'fearful', 'sad', 'happy', 'neutral']

    def predict(self, input_tensor):
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
        return output
