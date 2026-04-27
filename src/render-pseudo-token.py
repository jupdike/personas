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
    p.add_argument("--token-file-b", default=None,
                   help="Optional second token file. When set, renders a blend "
                        "sweep between --token-file (A) and --token-file-b (B).")
    p.add_argument("--blend-steps", type=int, default=4,
                   help="Number of intervals in the blend sweep. blend-steps=4 "
                        "produces 5 images per prompt at alpha = 0, 1/4, 2/4, 3/4, 1.")
    p.add_argument("--blend-mode", choices=["lerp", "slerp"], default="lerp",
                   help="lerp: naive (1-a)*A + a*B. slerp: spherical interpolation "
                        "on direction with linear interpolation of magnitude — useful "
                        "to test whether direction or magnitude is the load-bearing "
                        "axis of CLIP's embedding space.")
    p.add_argument("--blend-tok-name", default="<blend>",
                   help="Internal placeholder name to register for the blended row(s).")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Scalar multiplier applied to all rows before splicing. "
                        "Use to attenuate an over-amplified token (try 0.6–0.85) "
                        "without retraining. Default 1.0.")
    return p.parse_args()


def resolve_seed(s: int) -> int:
    return random.randint(1, 2**31 - 1) if s == 0 else s


def load_prompts(path: str | None) -> list[str]:
    if path is None:
        return list(DEFAULT_VALIDATION_PROMPTS)
    lines = Path(path).read_text().splitlines()
    return [line.strip() for line in lines
            if line.strip() and not line.strip().startswith("#")]


def load_token_payload(path: str):
    """Returns (payload, embeddings [N, 768], subtokens [N], user_facing_name)."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    placeholder = payload["token"]
    if "embeddings" in payload:
        embeddings = payload["embeddings"].float()
        subtokens = payload["subtokens"]
    else:
        embeddings = payload["embedding"].float().unsqueeze(0)
        subtokens = [placeholder]
    return payload, embeddings, subtokens, placeholder


def slerp(a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
    """Per-row slerp on direction; linear interpolation of magnitude. Shapes [N, D]."""
    norm_a = a.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    norm_b = b.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    ua, ub = a / norm_a, b / norm_b
    dot = (ua * ub).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.arccos(dot)
    sin_theta = torch.sin(theta)
    direction = torch.where(
        sin_theta < 1e-6,
        (1 - alpha) * ua + alpha * ub,  # near-collinear fallback
        (torch.sin((1 - alpha) * theta) / sin_theta) * ua
        + (torch.sin(alpha * theta) / sin_theta) * ub,
    )
    magnitude = (1 - alpha) * norm_a + alpha * norm_b
    return direction * magnitude


def blend_embeddings(a: torch.Tensor, b: torch.Tensor,
                     alpha: float, mode: str) -> torch.Tensor:
    if mode == "lerp":
        return (1 - alpha) * a + alpha * b
    if mode == "slerp":
        return slerp(a, b, alpha)
    raise ValueError(f"unknown blend mode: {mode!r}")


def main():
    args = parse_args()
    device = "mps"
    dtype = torch.float32 if args.dtype == "fp32" else torch.float16

    payload_a, emb_a, sub_a, name_a = load_token_payload(args.token_file)
    blend_mode_on = args.token_file_b is not None
    if blend_mode_on:
        payload_b, emb_b, sub_b, name_b = load_token_payload(args.token_file_b)
        if emb_a.shape != emb_b.shape:
            raise SystemExit(
                f"Token shape mismatch: A={tuple(emb_a.shape)} vs "
                f"B={tuple(emb_b.shape)}. Both files must have the same num_tokens."
            )
        # Register a fresh synthetic placeholder for the blended row(s).
        base = args.blend_tok_name.strip("<>")
        n_rows = emb_a.shape[0]
        if n_rows == 1:
            subtokens = [args.blend_tok_name]
        else:
            subtokens = [f"<{base}__{i}>" for i in range(n_rows)]
        placeholder = args.blend_tok_name
    else:
        payload_b = emb_b = sub_b = name_b = None
        subtokens = sub_a
        placeholder = name_a
    placeholder_run = " ".join(subtokens)

    saved_ckpt = payload_a.get("checkpoint_name")
    if saved_ckpt and saved_ckpt != model_name:
        print(f"<> WARNING: token was trained against {saved_ckpt!r} but current "
              f"checkpoint is {model_name!r}. Pseudo-tokens are coupled to U-Net "
              "weights — results may be poor.")
    row_norms_a = [f"{emb_a[i].norm().item():.3f}" for i in range(emb_a.shape[0])]
    print(f"<> Token A: {name_a} (||emb||={row_norms_a}, "
          f"source: {payload_a.get('source_prompt', '?')!r})")
    if blend_mode_on:
        row_norms_b = [f"{emb_b[i].norm().item():.3f}" for i in range(emb_b.shape[0])]
        print(f"<> Token B: {name_b} (||emb||={row_norms_b}, "
              f"source: {payload_b.get('source_prompt', '?')!r})")
        print(f"<> Blend mode: {args.blend_mode}, {args.blend_steps + 1} "
              f"steps, registered as {subtokens}")

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
    # mean_resizing=False — we overwrite the new rows anyway, so the multivariate-
    # normal init is wasted work and just emits a warning.
    text_encoder.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    placeholder_ids = tokenizer.convert_tokens_to_ids(subtokens)
    embedding_layer = text_encoder.get_input_embeddings()

    def splice(rows: torch.Tensor) -> None:
        if args.scale != 1.0:
            rows = rows * args.scale
        with torch.no_grad():
            embedding_layer.weight.data[placeholder_ids] = rows.to(
                embedding_layer.weight.dtype
            ).to(device)

    if not blend_mode_on:
        splice(emb_a)
        print(f"<> Spliced {emb_a.shape[0]} learned row(s) into ids {placeholder_ids}.")

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

    if blend_mode_on:
        alphas = [i / args.blend_steps for i in range(args.blend_steps + 1)]
        a_stem = Path(args.token_file).stem
        b_stem = Path(args.token_file_b).stem
    else:
        alphas = [None]
        a_stem = b_stem = None

    print(f"<> Rendering {len(prompts)} prompts × {args.n} images × "
          f"{len(alphas)} alpha(s) "
          f"({'img2img from ' + args.init if args.init else 'txt2img'}) ...")
    for prompt in prompts:
        print(f"=== {prompt}")
        for i in range(args.n):
            seed = resolve_seed(args.seed) + (i if args.seed != 0 else 0)
            for alpha in alphas:
                if alpha is not None:
                    splice(blend_embeddings(emb_a, emb_b, alpha, args.blend_mode))
                # Reset RNG for each alpha so only the embedding differs.
                generator = torch.Generator(device="cpu").manual_seed(seed)
                image = generate(generator, prompt)

                scale_meta = (f", scale={args.scale:.3f}" if args.scale != 1.0 else "")
                token_meta = (
                    f"Pseudo-token: {placeholder} "
                    f"(blend {a_stem} -> {b_stem}, mode={args.blend_mode}, "
                    f"alpha={alpha:.3f}{scale_meta}); "
                    if alpha is not None else
                    f"Pseudo-token: {placeholder} (file={args.token_file}{scale_meta}); "
                )
                parameters = (
                    f"Include in Image: {prompt}; "
                    f"Exclude from Image: {negative_prompt}; "
                    + token_meta
                    + f"Model: {model_name}; Steps: {args.steps}; "
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
                if alpha is not None:
                    stem = f"{a_stem}-{b_stem}-a{int(round(alpha * 100)):03d}"
                else:
                    stem = placeholder.strip("<>").replace(":", "_")
                if args.scale != 1.0:
                    stem = f"{stem}-s{int(round(args.scale * 100)):03d}"
                if stem_kind:
                    stem = f"{stem_kind}-{stem}"
                path = out_dir / timestamped_filename(f"{stem}-{seed}")
                image.save(path, pnginfo=png_meta)
                tag = f"  α={alpha:.2f}" if alpha is not None else ""
                print(f"   {tag} -> {path}")


if __name__ == "__main__":
    main()
