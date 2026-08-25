import torch
import torch.nn as nn
from torchvision import models

def get_mobilenetv2_model(num_classes=38):
    """
    Membangun model MobileNetV2 dengan classifier head yang dimodifikasi.
    Sesuai dengan arsitektur pada inference script.
    """
    model = models.mobilenet_v2(weights=None)  # Tidak pakai pretrained
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, num_classes)
    )
    return model