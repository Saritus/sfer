import os.path
import urllib.request
from enum import Enum

import numpy


class PrebuiltModel(Enum):
    EfficientNetB0 = "EfficientNetB0"
    EfficientNetB1 = "EfficientNetB1"
    EfficientNetB2 = "EfficientNetB2"
    DenseNet121 = "DenseNet121"
    MobileNetV2 = "MobileNetV2"
    MobileNet = "MobileNet"
    NASNetMobile = "NASNetMobile"
    ResNet18 = "ResNet18"
    ResNet34 = "ResNet34"
    ResNet50 = "ResNet50"
    SEResNet18 = "SEResNet18"
    SEResNet34 = "SEResNet34"
    SEResNet50 = "SEResNet50"
    VGG13 = "VGG13"
    DeXpression = "DeXpression"
    LeNet = "LeNet"


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def download_model(model: PrebuiltModel, img_size: int) -> str:
    """Download model from GitHub"""
    filename = f"{model.value}_{img_size}x{img_size}.hdf5"
    url = f"https://github.com/Saritus/sfer/releases/download/0.0.1/{filename}"
    filepath = os.path.join("models", filename)
    if not os.path.exists(filepath):
        print("Downloading model...")
        ensure_dir("models")
        urllib.request.urlretrieve(url, filepath)
        print("Model downloaded.")
    return filepath


def download_emotions() -> str:
    """Download model from GitHub"""
    filename = "emotions.pkl"
    url = f"https://github.com/Saritus/sfer/releases/download/0.0.1/{filename}"
    filepath = os.path.join("models", filename)
    if not os.path.exists(filepath):
        ensure_dir("models")
        urllib.request.urlretrieve(url, filepath)
    return filepath
