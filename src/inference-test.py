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

prompt = open('test-prompt.txt').read().strip()
N=1

for i in range(N):
    image = pipe(prompt, num_inference_steps=25, num_images_per_prompt=1, negative_prompt="nsfw").images[0]
    image.save(f"output/{timestamped_filename(f'out-{i}')}")
