import argparse

import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from PIL import Image

from util import timestamped_filename

EDGE_OF_REALISM_MODEL_PATH = "models/edgeOfRealism_eorV20BakedVAE.safetensors"

parser = argparse.ArgumentParser()
parser.add_argument("--init", help="Path to init image for img2img mode")
parser.add_argument("--steps", type=int, default=25, help="Number of inference steps")
parser.add_argument("--strength", type=float, default=0.6,
                    help="img2img strength: 0.0 = identity, 1.0 = ignore init image")
parser.add_argument("-n", type=int, default=1, help="Number of images to generate")
args = parser.parse_args()

pipe = StableDiffusionPipeline.from_single_file(
    EDGE_OF_REALISM_MODEL_PATH,
    config="stable-diffusion-v1-5/stable-diffusion-v1-5",
    safety_checker=None,
    requires_safety_checker=False,
    torch_dtype=torch.float32,  # fp16 is flaky on MPS — stick to fp32 initially
)
pipe = pipe.to("mps")

prompt = open('test-prompt.txt').read().strip()
negative_prompt = open('neg-prompt.txt').read().strip()

if args.init:
    img2img = StableDiffusionImg2ImgPipeline(**pipe.components)
    init_image = Image.open(args.init).convert("RGB").resize((512, 512))
    stem = "i2i"
    def generate():
        return img2img(
            prompt=prompt,
            guidance_scale=7.5,
            image=init_image,
            strength=args.strength,
            safety_checker=None,
            requires_safety_checker=False,
            num_inference_steps=args.steps,
            negative_prompt=negative_prompt,
        ).images[0]
else:
    stem = "out"
    def generate():
        return pipe(
            prompt,
            guidance_scale=7.5,
            num_inference_steps=args.steps,
            num_images_per_prompt=1,
            negative_prompt=negative_prompt,
        ).images[0]

for i in range(args.n):
    image = generate()
    image.save(f"output/{timestamped_filename(f'{stem}-{i}')}")
