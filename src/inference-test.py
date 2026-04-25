import argparse
import random

import torch
from diffusers import (
    DPMSolverMultistepScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
)
from PIL import Image, PngImagePlugin

from util import timestamped_filename, last_path_component, xmp_description_packet

EDGE_OF_REALISM_MODEL_PATH = "models/edgeOfRealism_eorV20BakedVAE.safetensors"
model_name = last_path_component(EDGE_OF_REALISM_MODEL_PATH)

EPIC_PREALISM_MODEL_PATH = "models/epicrealism_naturalSinRC1VAE.safetensors"
model_name = last_path_component(EPIC_PREALISM_MODEL_PATH)

model_path = EPIC_PREALISM_MODEL_PATH

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
    model_path,
    config="stable-diffusion-v1-5/stable-diffusion-v1-5",
    safety_checker=None,
    requires_safety_checker=False,
    torch_dtype=torch.float32,  # fp16 is flaky on MPS — stick to fp32 initially
)
pipe = pipe.to("mps")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    use_karras_sigmas=True,
    algorithm_type="dpmsolver++",
)

prompt = open('test-prompt.txt').read().strip()
negative_prompt = open('neg-prompt.txt').read().strip()

if args.init:
    img2img = StableDiffusionImg2ImgPipeline(
        **pipe.components,
        requires_safety_checker=False,
    )
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
            generator=generator,
        ).images[0]

for i in range(args.n):
    seed = resolve_seed(args.seed) + (i if args.seed != 0 else 0)
    parameters = f"Include in Image: {prompt}; Exclude from Image: {negative_prompt}; Model: {model_name}; Steps: {args.steps}; Guidance Scale: {args.guidance}; Seed: {seed}; Size: 512x512; Scheduler: DPM-Solver++; ML Compute Unit: MPS; Generator: Personas 0.1 + HuggingFace diffusers"
    generator = torch.Generator(device="cpu").manual_seed(seed)
    image = generate(generator)
    png_meta = PngImagePlugin.PngInfo()
    png_meta.add_text("Description", parameters)
    png_meta.add_text("parameters", parameters)
    png_meta.add_text("Software", "Personas 0.1 + HuggingFace diffusers")
    png_meta.add_itxt("XML:com.adobe.xmp", xmp_description_packet(parameters), lang="", tkey="")
    image.save(f"output/{timestamped_filename(f'{stem}-{seed}')}", pnginfo=png_meta)
