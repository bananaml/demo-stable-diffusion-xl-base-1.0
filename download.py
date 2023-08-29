from diffusers import DiffusionPipeline
import torch

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

def download_model() -> tuple:
    """Download the model"""
    model = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

if __name__ == "__main__":
    download_model()
    