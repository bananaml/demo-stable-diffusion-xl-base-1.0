from diffusers import DiffusionPipeline
import torch

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

def download_model() -> tuple:
    """Download the model"""
    model = DiffusionPipeline.from_pretrained(MODEL, use_safetensors=True)


if __name__ == "__main__":
    download_model()
    