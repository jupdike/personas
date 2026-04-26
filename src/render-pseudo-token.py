"""Render images with a trained pseudo-token spliced into the CLIP embedding table.

Loads a `.pt` produced by train-pseudo-token.py, registers the placeholder string
with the tokenizer, writes the learned 768-vector into the placeholder row, and
runs Stable Diffusion on a list of prompts containing that placeholder.

Defaults to the validation prompt set described in PLAN.md.
"""
import argparse
import random
from pathlib import Path

import torch
from diffusers import (
    DPMSolverMultistepScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
)
from PIL import Image, PngImagePlugin

from models import experiment_path, model_name, model_path
from util import timestamped_filename, xmp_description_packet


# Default validation prompts. {tok} is replaced with the placeholder string.
DEFAULT_VALIDATION_PROMPTS = [
    # Solo fidelity
    "a photo of {tok}",
    "portrait of {tok}",
    # Novel scene composition
    "{tok}, medieval armor, castle courtyard",
    "{tok}, reading a book, library interior",
    "{tok}, hiking on a mountain trail, golden hour",
    # Modifier compatibility
    "{tok}, in the style of Vermeer",
    "{tok}, pencil sketch, textured paper",
    # Token-budget check
    "{tok}, sitting at a cafe, autumn leaves falling, golden afternoon light, "
    "holding a book, shallow depth of field, film grain",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--token-file", required=True,
                   help="Path to the .pt produced by train-pseudo-token.py")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=0,
                   help="Base seed; 0 = random per image")
    p.add_argument("-n", type=int, default=1, help="Images per prompt")
    p.add_argument("--prompts-file", default=None,
                   help="Optional path to a text file of prompts (one per line, "
                        "use {tok} placeholder). Lines starting with # are ignored.")
    p.add_argument("--dtype", choices=["fp32", "fp16"], default="fp16")
    p.add_argument("--out-dir", default="output",
                   help="Directory to save rendered PNGs")
    p.add_argument("--neg-prompt-file", default="neg-prompt.txt")
    p.add_argument("--init", default=None,
                   help="Optional init image path. When set, renders in img2img mode.")
    p.add_argument("--strength", type=float, default=0.6,
                   help="img2img strength: 0.0 = identity, 1.0 = ignore init image")
    return p.parse_args()


def resolve_seed(s: int) -> int:
    return random.randint(1, 2**31 - 1) if s == 0 else s


def load_prompts(path: str | None) -> list[str]:
    if path is None:
        return list(DEFAULT_VALIDATION_PROMPTS)
    lines = Path(path).read_text().splitlines()
    return [line.strip() for line in lines
            if line.strip() and not line.strip().startswith("#")]


def main():
    args = parse_args()
    device = "mps"
    dtype = torch.float32 if args.dtype == "fp32" else torch.float16

    payload = torch.load(args.token_file, map_location="cpu", weights_only=False)
    placeholder = payload["token"]  # user-facing name (e.g. "<pref_003>")
    if "embeddings" in payload:
        embeddings = payload["embeddings"].float()  # [N, 768]
        subtokens = payload["subtokens"]
    else:
        # Legacy single-token format.
        embeddings = payload["embedding"].float().unsqueeze(0)  # [1, 768]
        subtokens = [placeholder]
    placeholder_run = " ".join(subtokens)

    saved_ckpt = payload.get("checkpoint_name")
    if saved_ckpt and saved_ckpt != model_name:
        print(f"<> WARNING: token was trained against {saved_ckpt!r} but current "
              f"checkpoint is {model_name!r}. Pseudo-tokens are coupled to U-Net "
              "weights — results may be poor.")
    row_norms = [f"{embeddings[i].norm().item():.3f}" for i in range(embeddings.shape[0])]
    print(f"<> Token: {placeholder} (subtokens={subtokens}, ||emb||={row_norms})")
    print(f"<> Trained against: {saved_ckpt}; source prompt: "
          f"{payload.get('source_prompt', '?')!r}")

    print("<> Loading pipeline ...")
    pipe = StableDiffusionPipeline.from_single_file(
        model_path,
        config="stable-diffusion-v1-5/stable-diffusion-v1-5",
        safety_checker=None,
        requires_safety_checker=False,
        torch_dtype=dtype,
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="dpmsolver++",
    )

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    num_added = tokenizer.add_tokens(subtokens)
    if num_added != len(subtokens):
        raise SystemExit(
            f"One or more of {subtokens!r} already exist in the tokenizer."
        )
    text_encoder.resize_token_embeddings(len(tokenizer))
    placeholder_ids = tokenizer.convert_tokens_to_ids(subtokens)
    embedding_layer = text_encoder.get_input_embeddings()
    with torch.no_grad():
        embedding_layer.weight.data[placeholder_ids] = embeddings.to(
            embedding_layer.weight.dtype
        ).to(device)
    print(f"<> Spliced {embeddings.shape[0]} learned row(s) into ids {placeholder_ids}.")

    prompt_templates = load_prompts(args.prompts_file)
    # Templates use {tok} (the user-facing name). Substitute it, then expand the
    # user-facing name to the registered subtoken run before tokenization.
    prompts = [
        t.format(tok=placeholder).replace(placeholder, placeholder_run)
        for t in prompt_templates
    ]
    negative_prompt = Path(args.neg_prompt_file).read_text().strip() \
        if Path(args.neg_prompt_file).exists() else ""

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.init:
        img2img = StableDiffusionImg2ImgPipeline(
            **pipe.components,
            requires_safety_checker=False,
        )
        init_image = Image.open(args.init).convert("RGB").resize((512, 512))
        stem_kind = "i2i"

        def generate(generator, p):
            return img2img(
                p,
                image=init_image,
                strength=args.strength,
                guidance_scale=args.guidance,
                num_inference_steps=args.steps,
                negative_prompt=negative_prompt,
                generator=generator,
            ).images[0]
    else:
        stem_kind = None

        def generate(generator, p):
            return pipe(
                p,
                guidance_scale=args.guidance,
                num_inference_steps=args.steps,
                num_images_per_prompt=1,
                negative_prompt=negative_prompt,
                generator=generator,
            ).images[0]

    print(f"<> Rendering {len(prompts)} prompts × {args.n} images "
          f"({'img2img from ' + args.init if args.init else 'txt2img'}) ...")
    for prompt in prompts:
        print(f"=== {prompt}")
        for i in range(args.n):
            seed = resolve_seed(args.seed) + (i if args.seed != 0 else 0)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            image = generate(generator, prompt)

            parameters = (
                f"Include in Image: {prompt}; "
                f"Exclude from Image: {negative_prompt}; "
                f"Pseudo-token: {placeholder} (file={args.token_file}); "
                f"Model: {model_name}; Steps: {args.steps}; "
                f"Guidance Scale: {args.guidance}; Seed: {seed}; "
                + (f"Init Image: {args.init}; Strength: {args.strength}; "
                   if args.init else "")
                + f"Size: 512x512; Scheduler: DPM-Solver++; "
                f"Generator: Personas 0.1 + HuggingFace diffusers"
            )
            png_meta = PngImagePlugin.PngInfo()
            png_meta.add_text("Description", parameters)
            png_meta.add_text("parameters", parameters)
            png_meta.add_text("Software", "Personas 0.1 + HuggingFace diffusers")
            png_meta.add_itxt(
                "XML:com.adobe.xmp",
                xmp_description_packet(parameters),
                lang="", tkey="",
            )
            stem = placeholder.strip("<>").replace(":", "_")
            if stem_kind:
                stem = f"{stem_kind}-{stem}"
            path = out_dir / timestamped_filename(f"{stem}-{seed}")
            image.save(path, pnginfo=png_meta)
            print(f"    -> {path}")


if __name__ == "__main__":
    main()
