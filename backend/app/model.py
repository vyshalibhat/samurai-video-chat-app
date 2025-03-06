# backend/app/model.py
# (Optional) Could unify or just remove this file if not needed.

import os
import torch
import torch.nn as nn
import torchvision.models.video as models

class EmotionResNet3DLocal:
    def __init__(self, model_path="6emotions_resnet3dV2.pth"):
        base_dir = os.path.dirname(__file__)
        full_model_path = os.path.join(base_dir, model_path)

