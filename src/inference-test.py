import argparse
import random

import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from PIL import Image

from util import timestamped_filename

EDGE_OF_REALISM_MODEL_PATH = "models/edgeOfRealism_eorV20BakedVAE.safetensors"

parser = argparse.ArgumentParser()
parser.add_argument("--init", help="Path to init image for img2img mode")
parser.add_argument("--steps", type=int, default=25, help="Number of inference steps")
parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility (0 = pick randomly)")
parser.add_argument("--guidance", type=float, default=7.5, help="Classifier-free guidance scale")
parser.add_argument("--strength", type=float, default=0.6,
                    help="img2img strength: 0.0 = identity, 1.0 = ignore init image")
parser.add_argument("-n", type=int, default=1, help="Number of images to generate")
args = parser.parse_args()


def resolve_seed(s: int) -> int:
    return random.randint(1, 2**31 - 1) if s == 0 else s


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
    def generate(generator):
        return img2img(
            prompt=prompt,
            image=init_image,
            strength=args.strength,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            negative_prompt=negative_prompt,
            safety_checker=None,
            requires_safety_checker=False,
            generator=generator,
        ).images[0]
else:
    stem = "out"
    def generate(generator):
        return pipe(
            prompt,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            num_images_per_prompt=1,
            negative_prompt=negative_prompt,
            safety_checker=None,
            requires_safety_checker=False,
            generator=generator,
        ).images[0]

for i in range(args.n):
    seed = resolve_seed(args.seed) + (i if args.seed != 0 else 0)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    image = generate(generator)
    image.save(f"output/{timestamped_filename(f'{stem}-{seed}')}")
