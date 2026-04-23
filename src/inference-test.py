import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_single_file(
    "models/edgeOfRealism_eorV20BakedVAE.safetensors",
    config="stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float32,  # fp16 is flaky on MPS — stick to fp32 initially
)
pipe = pipe.to("mps")

image = pipe("a portrait of a woman", num_inference_steps=25).images[0]
image.save("output/out.png")
