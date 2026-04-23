import torch
from diffusers import StableDiffusionPipeline

from util import timestamped_filename

EDGE_OF_REALISM_MODEL_PATH = "models/edgeOfRealism_eorV20BakedVAE.safetensors"

pipe = StableDiffusionPipeline.from_single_file(
    EDGE_OF_REALISM_MODEL_PATH,
    config="stable-diffusion-v1-5/stable-diffusion-v1-5",
    safety_checker=None,
    requires_safety_checker=False,
    torch_dtype=torch.float32,  # fp16 is flaky on MPS — stick to fp32 initially
)
pipe = pipe.to("mps")

image = pipe("a portrait of a woman", num_inference_steps=25).images[0]
image.save(f"output/{timestamped_filename('out')}")
