from potassium import Potassium, Request, Response
from diffusers import DiffusionPipeline
import torch
import base64
from io import BytesIO

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

app = Potassium("stable-diffusion-xl-base-1.0")

@app.init
def init() -> dict:
    """Initialize the application with the model."""
    model = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    model.to("cuda")
    return {
        "model": model
    }

@app.handler()
def handler(context: dict, request: Request) -> Response:
    """Handle a request to generate image from a prompt."""
    model = context.get("model")
    prompt = request.json.get("prompt")
    images = model(prompt=prompt).images[0]
    buffered = BytesIO()
    images.save(buffered, format="JPEG", quality=80)
    img_str = base64.b64encode(buffered.getvalue())
    return Response(json={"output": str(img_str, "utf-8")}, status=200)

if __name__ == "__main__":
    app.serve()