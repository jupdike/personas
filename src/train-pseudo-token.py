"""Train a multi-row CLIP pseudo-token, either by distilling a source prompt
(--mode distill, default) or by reconstructing reference images
(--mode images, classical Textual Inversion).

distill: reads {experiment_path}/persona.txt as the source prompt and matches
the U-Net's noise prediction on the placeholder to its prediction on the
source. No images required.

images: reads {ref_dir}/*.{png,jpg,jpeg,webp} as identity references and
trains the placeholder so the U-Net can denoise their VAE latents when
conditioned on the placeholder. Supervision is real-noise reconstruction at
random timesteps — the standard TI loss.

Writes: {experiment_path}/{placeholder-stripped}.pt + .json metadata sidecar.
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, StableDiffusionPipeline
from PIL import Image
from torchvision import transforms as T

from models import experiment_path, model_name, model_path

SCENE_TEMPLATES = [
    "{s}",
    # Photography / Lighting
    "portrait of {s}, at the beach, golden hour",
    "portrait of {s}, studio lighting, neutral backdrop",
    "portrait of {s}, soft window light, cafe interior",
    "closeup of {s}, dramatic side lighting",
    "{s}, urban street, casual outfit",
    "{s}, autumn forest, diffused light",
    "black and white photo of {s}, high contrast",
    "{s}, rainy evening, moody reflections",
    "{s}, natural outdoor lighting",
    "a photo of {s}",
    "portrait of {s}",
    "{s}, overcast sky, open field",
    "{s}, candlelight, dark interior",
    "{s}, morning mist, countryside",
    "closeup of {s}, soft bokeh background",
    "{s}, neon lights, night city",
    "{s}, harsh midday sun, desert",
    "{s}, golden hour, rooftop",
    "{s}, foggy street, shallow depth of field",
    "{s}, backlit by window, warm tones",
    # Settings / Context
    "{s}, library interior, reading",
    "{s}, coffee shop, busy background",
    "{s}, mountain trail, overcast",
    "{s}, snowy landscape, winter clothing",
    "{s}, poolside, summer afternoon",
    "{s}, market street, candid shot",
    "{s}, empty subway car",
    "{s}, concert crowd, stage lights",
    "{s}, standing in rain, wet street",
    "{s}, rooftop garden, dusk",
    # Styled / Artistic
    "oil painting of {s}, soft brushwork",
    "watercolor portrait of {s}, loose style",
    "charcoal sketch of {s}, textured paper",
    "vintage photograph of {s}, sepia tones",
    "cinematic still of {s}, anamorphic lens",
    "editorial photo of {s}, fashion magazine",
    "film grain photo of {s}, 35mm",
    "renaissance painting style portrait of {s}",
    "anime style portrait of {s}",
    "noir photo of {s}, low key lighting",
    # Pose / Framing
    "full body shot of {s}, plain background",
    "side profile of {s}, neutral expression",
    "overhead angle of {s}, looking up",
    "{s}, looking over shoulder, outdoor",
    "{s}, arms crossed, confident pose",
    "{s}, candid laugh, outdoor setting",
    "{s}, seated, contemplative expression",
    "{s}, walking away, motion blur",
    "extreme closeup of {s}, eyes in focus",
    "{s}, silhouette against sunset sky",
]


# For --mode images: keep the textual context neutral. Scene-specific wrappers
# would tie identity supervision to scenes that are not in the reference images
# — exactly the leakage failure mode we're trying to leave behind.
IMAGE_TEMPLATES = [
    "{s}",
    "a photo of {s}",
    "a portrait of {s}",
    "a closeup of {s}",
    "a headshot of {s}",
    "a portrait photo of {s}",
    "a photograph of {s}",
    "a picture of {s}",
    "a photo of the person {s}",
    "a portrait of the person {s}",
]


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["distill", "images"], default="distill",
                   help="distill: match U-Net noise prediction to the source prompt "
                        "in persona.txt (no images needed). images: classical TI — "
                        "reconstruct reference images from --ref-dir.")
    p.add_argument("--ref-dir", default=None,
                   help="Directory of reference images for --mode=images. "
                        "Reads *.png, *.jpg, *.jpeg, *.webp.")
    p.add_argument("--placeholder", default="<pref_001>",
                   help="User-facing placeholder token. With --num-tokens > 1, "
                        "internal subtokens like <name__0>, <name__1>, ... are "
                        "registered and the user-facing name expands to that run "
                        "at training and inference time.")
    p.add_argument("--num-tokens", type=int, default=1,
                   help="Number of pseudo-token rows to train. >1 gives the "
                        "transformer multiple positions to encode the persona.")
    p.add_argument("--steps", type=int, default=None,
                   help="Training steps. Defaults: 2000 for distill, 3000 for images.")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init-mode", choices=["persona", "words"], default=None,
                   help="persona: initialize from the mean of the persona's own token "
                        "embeddings. words: mean of --init-words. Defaults: persona "
                        "for distill mode (when persona.txt exists); words for images.")
    p.add_argument("--init-words", default="woman,portrait,person",
                   help="Used only when --init-mode=words.")
    p.add_argument("--timestep-bias", choices=["uniform", "identity"], default="identity",
                   help="uniform: sample t over [0, T). identity: 70%% from [300, 900] "
                        "where text most shapes the image, 30%% from full range.")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--save-every", type=int, default=500,
                   help="Save a checkpoint every N steps (0 to disable)")
    p.add_argument("--dtype", choices=["fp32", "fp16"], default="fp16",
                   help="Computation dtype for frozen weights. fp16 with an fp32 master "
                        "copy of the trainable row (and fp32 loss) is the default — "
                        "much faster on MPS and stable.")
    p.add_argument("--out", default=None,
                   help="Override output path. Defaults to "
                        "{experiment_path}/{placeholder-stripped}.pt")
    return p.parse_args()


def preencode_references(ref_dir: str, vae, device: str, dtype: torch.dtype):
    """Load images, encode them through the VAE to latents [K, 4, 64, 64].

    Returns (latents on device, list of (Path, sha256) tuples).
    """
    paths = sorted([
        p for p in Path(ref_dir).iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    ])
    if not paths:
        raise SystemExit(f"No images found in {ref_dir!r}")

    transform = T.Compose([
        T.Resize(512, interpolation=T.InterpolationMode.LANCZOS),
        T.CenterCrop(512),
        T.ToTensor(),  # -> [3, 512, 512] in [0, 1]
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1, 1]
    ])

    latents = []
    digests = []
    with torch.no_grad():
        for p in paths:
            img = Image.open(p).convert("RGB")
            pixel = transform(img).unsqueeze(0).to(device=device, dtype=dtype)
            lat = vae.encode(pixel).latent_dist.sample()
            lat = lat * vae.config.scaling_factor  # standard SD 1.5 latent scale
            latents.append(lat)
            digests.append((p, file_sha256(str(p))[:16]))
    return torch.cat(latents, dim=0), digests


def main():
    args = parse_args()

    if args.num_tokens < 1:
        raise SystemExit("--num-tokens must be >= 1")
    if args.mode == "images" and not args.ref_dir:
        raise SystemExit("--mode=images requires --ref-dir")
    if args.steps is None:
        args.steps = 3000 if args.mode == "images" else 2000

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = "mps"
    dtype = torch.float32 if args.dtype == "fp32" else torch.float16

    # persona.txt: required for distill mode (it IS the source prompt). Optional
    # in images mode — kept around for human-readable description and as the
    # source pool for --init-mode=persona.
    persona_path = f"{experiment_path}/persona.txt"
    if Path(persona_path).exists():
        persona = open(persona_path).read().strip()
    elif args.mode == "distill":
        raise SystemExit(f"distill mode requires {persona_path}")
    else:
        persona = ""

    # Default --init-mode based on training mode and what's available.
    if args.init_mode is None:
        if args.mode == "distill":
            args.init_mode = "persona"
        else:
            args.init_mode = "persona" if persona else "words"

    print(f"<> Model: {model_name}")
    print(f"<> Experiment path: {experiment_path}")
    print(f"<> Mode: {args.mode}"
          + (f"  ref-dir: {args.ref_dir}" if args.mode == "images" else ""))
    if persona:
        print(f"<> Persona: {persona}")
    print(f"<> Placeholder: {args.placeholder}")
    print(f"<> Steps: {args.steps}  lr: {args.lr}  dtype: {args.dtype}  "
          f"init: {args.init_mode}  timestep-bias: {args.timestep_bias}  "
          f"num-tokens: {args.num_tokens}")

    print("<> Loading pipeline ...")
    pipe = StableDiffusionPipeline.from_single_file(
        model_path,
        config="stable-diffusion-v1-5/stable-diffusion-v1-5",
        safety_checker=None,
        requires_safety_checker=False,
        torch_dtype=dtype,
    ).to(device)

    text_encoder = pipe.text_encoder
    unet = pipe.unet
    tokenizer = pipe.tokenizer

    # In images mode, encode references once with the VAE, then drop it. In
    # distill mode the VAE is never needed.
    ref_latents = None
    ref_digests = None
    if args.mode == "images":
        print(f"<> Encoding references in {args.ref_dir} ...")
        ref_latents, ref_digests = preencode_references(
            args.ref_dir, pipe.vae, device, dtype
        )
        print(f"<> Encoded {ref_latents.shape[0]} reference latents "
              f"(shape {tuple(ref_latents.shape)})")
    pipe.vae = None  # free VAE memory once references are cached.

    # Build the subtoken list. With num_tokens=1 we use the placeholder name itself
    # as the only registered token (preserves the prior single-token format/UX).
    if args.num_tokens == 1:
        subtokens = [args.placeholder]
    else:
        base = args.placeholder.strip("<>")
        subtokens = [f"<{base}__{i}>" for i in range(args.num_tokens)]
    placeholder_run = " ".join(subtokens)

    # Add the subtokens to the tokenizer and resize the embedding table.
    num_added = tokenizer.add_tokens(subtokens)
    if num_added != len(subtokens):
        raise SystemExit(
            f"One or more of {subtokens!r} already exist in the tokenizer; "
            "pick a different --placeholder."
        )
    text_encoder.resize_token_embeddings(len(tokenizer))
    placeholder_ids = tokenizer.convert_tokens_to_ids(subtokens)
    print(f"<> Subtokens: {subtokens} -> ids {placeholder_ids}")
    print(f"<> Placeholder run substituted into templates: {placeholder_run!r}")

    embedding_layer = text_encoder.get_input_embeddings()
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    # Build the source pool of token ids to initialize from.
    if args.init_mode == "persona":
        persona_ids = tokenizer(persona, add_special_tokens=False).input_ids
        persona_ids = [i for i in persona_ids
                       if i not in (bos_id, eos_id, pad_id) and i not in placeholder_ids]
        if not persona_ids:
            raise SystemExit("Persona tokenized to zero content tokens; cannot init.")
        init_pool = persona_ids
        init_label = f"persona ({len(init_pool)} tokens)"
    else:
        words = [w.strip() for w in args.init_words.split(",") if w.strip()]
        init_pool = tokenizer.convert_tokens_to_ids(words)
        init_label = f"words {words}"

    # Init each row from a different chunk of the init pool (rotating if pool is
    # smaller than num_tokens). Distinct starting points let the rows specialize.
    with torch.no_grad():
        for i, pid in enumerate(placeholder_ids):
            if args.num_tokens == 1:
                chunk = init_pool
            else:
                # Split init_pool into N contiguous chunks; if pool < N, repeat.
                chunk_size = max(1, len(init_pool) // args.num_tokens)
                start = (i * chunk_size) % len(init_pool)
                end = start + chunk_size
                chunk = init_pool[start:end] if end <= len(init_pool) else (
                    init_pool[start:] + init_pool[:end - len(init_pool)]
                )
            init_vec = embedding_layer.weight[chunk].float().mean(0)
            embedding_layer.weight[pid] = init_vec.to(embedding_layer.weight.dtype)
    init_norms = [
        embedding_layer.weight[pid].float().norm().item() for pid in placeholder_ids
    ]
    print(f"<> Initialized from {init_label}; per-row ||emb|| = "
          f"{[f'{n:.3f}' for n in init_norms]}")

    # Freeze everything. The embedding weight needs requires_grad=True so autograd
    # produces a gradient for our rows; we extract only the rows we care about.
    for p in text_encoder.parameters():
        p.requires_grad_(False)
    for p in unet.parameters():
        p.requires_grad_(False)
    embedding_layer.weight.requires_grad_(True)

    # fp32 master copy of the placeholder rows. The frozen pipeline stays in fp16
    # for speed/memory on MPS; the optimizer step runs in fp32 to avoid NaN from
    # underflow on tiny noise-prediction MSE gradients.
    master = nn.Parameter(
        embedding_layer.weight[placeholder_ids].detach().float().clone()
    )  # shape [N, 768]
    optimizer = torch.optim.AdamW([master], lr=args.lr)

    num_train_timesteps = pipe.scheduler.config.num_train_timesteps

    # In images mode we need a forward-process scheduler with a clean add_noise.
    # Build one explicitly so the result doesn't depend on whatever scheduler
    # the pipeline ships with.
    train_scheduler = (
        DDPMScheduler.from_config(pipe.scheduler.config)
        if args.mode == "images" else None
    )

    templates = SCENE_TEMPLATES if args.mode == "distill" else IMAGE_TEMPLATES

    losses = []
    t0 = time.time()
    last_log = t0
    print(f"<> Starting training for {args.steps} steps ...")
    for step in range(args.steps):
        # Push the current fp32 master rows into the (possibly-fp16) embedding table.
        with torch.no_grad():
            embedding_layer.weight.data[placeholder_ids] = master.data.to(
                embedding_layer.weight.dtype
            )

        template = random.choice(templates)
        pred_prompt = template.format(s=placeholder_run)
        pred_ids = tokenizer(
            pred_prompt, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)

        if args.timestep_bias == "identity" and random.random() < 0.7:
            # Identity-shaping band: text most influences the image around mid-noise.
            t = torch.randint(300, 900, (1,), device=device).long()
        else:
            t = torch.randint(0, num_train_timesteps, (1,), device=device).long()

        if args.mode == "distill":
            target_prompt = template.format(s=persona)
            target_ids = tokenizer(
                target_prompt, padding="max_length", max_length=77,
                truncation=True, return_tensors="pt",
            ).input_ids.to(device)
            z = torch.randn(1, 4, 64, 64, device=device, dtype=dtype)
            with torch.no_grad():
                target_hs = text_encoder(target_ids)[0]
                eps_target = unet(z, t, encoder_hidden_states=target_hs).sample
            pred_hs = text_encoder(pred_ids)[0]
            eps_pred = unet(z, t, encoder_hidden_states=pred_hs).sample
            target_signal = eps_target
        else:
            # Sample one cached reference latent and run the standard latent-
            # diffusion forward process to build a noisy input.
            ref_idx = random.randrange(ref_latents.shape[0])
            ref_lat = ref_latents[ref_idx:ref_idx + 1]  # [1, 4, 64, 64]
            noise = torch.randn_like(ref_lat)
            noisy = train_scheduler.add_noise(ref_lat, noise, t)
            pred_hs = text_encoder(pred_ids)[0]
            eps_pred = unet(noisy, t, encoder_hidden_states=pred_hs).sample
            target_signal = noise

        # Compute the loss in fp32 — noise-prediction MSE is small enough to
        # underflow in fp16 (we hit NaNs in earlier runs without this cast).
        loss = F.mse_loss(eps_pred.float(), target_signal.float())

        embedding_layer.weight.grad = None
        master.grad = None
        loss.backward()

        # Pull the gradient for our rows out of the (fp16) embedding table, cast to
        # fp32, and apply the AdamW step on the master copy.
        if embedding_layer.weight.grad is not None:
            master.grad = (
                embedding_layer.weight.grad[placeholder_ids].detach().float().clone()
            )
            if not torch.isfinite(master.grad).all():
                # Skip this step if anything blew up; keep training stable.
                master.grad = None
        optimizer.step()

        losses.append(loss.item())
        if step % args.log_every == 0 or step == args.steps - 1:
            window = losses[-args.log_every:] if losses else [0.0]
            avg = sum(window) / len(window)
            now = time.time()
            sps = args.log_every / max(1e-6, now - last_log) if step > 0 else 0.0
            last_log = now
            row_norms = [f"{master[i].detach().norm().item():.3f}"
                         for i in range(args.num_tokens)]
            print(f"step {step:5d}  loss={loss.item():.6f}  avg{args.log_every}={avg:.6f}  "
                  f"||emb||={row_norms}  ({sps:.2f} step/s)")

        if args.save_every and step > 0 and step % args.save_every == 0:
            save_token(args, persona, master.detach(), subtokens, step)

    elapsed = time.time() - t0
    print(f"<> Done in {elapsed:.1f}s ({args.steps / elapsed:.2f} step/s)")

    out_path = save_token(args, persona, master.detach(), subtokens, args.steps)
    print(f"<> Saved final pseudo-token to {out_path}")

    # Sidecar JSON for human inspection (the .pt itself is the source of truth).
    meta_path = out_path.replace(".pt", ".json")
    sidecar = {
        "mode": args.mode,
        "token": args.placeholder,
        "subtokens": subtokens,
        "num_tokens": args.num_tokens,
        "checkpoint_name": model_name,
        "checkpoint_path": model_path,
        "source_prompt": persona,
        "num_steps": args.steps,
        "lr": args.lr,
        "init_mode": args.init_mode,
        "init_words": (
            None if args.init_mode == "persona" else
            [w.strip() for w in args.init_words.split(",") if w.strip()]
        ),
        "timestep_bias": args.timestep_bias,
        "dtype": args.dtype,
        "seed": args.seed,
        "final_loss": losses[-1] if losses else None,
        "avg_loss_last_100": (
            sum(losses[-100:]) / max(1, len(losses[-100:])) if losses else None
        ),
    }
    if args.mode == "images":
        sidecar["ref_dir"] = args.ref_dir
        sidecar["ref_count"] = len(ref_digests)
        sidecar["refs"] = [{"path": str(p), "sha256_16": h} for p, h in ref_digests]
    with open(meta_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"<> Metadata sidecar at {meta_path}")


def save_token(args, persona: str, embeddings: torch.Tensor,
               subtokens: list[str], step: int) -> str:
    name = args.placeholder.strip("<>").replace(":", "_").replace("/", "_")
    if args.out:
        out = args.out
    else:
        out = f"{experiment_path}/{name}.pt"
    learned = embeddings.detach().float().cpu().clone()  # [N, 768]
    payload = {
        "token": args.placeholder,
        "subtokens": subtokens,
        "embeddings": learned,
        "checkpoint_hash": file_sha256(model_path),
        "checkpoint_name": model_name,
        "source_prompt": persona,
        "num_steps": step,
        "lr": args.lr,
    }
    # Backward-compat alias for single-token files (older render scripts read this).
    if learned.shape[0] == 1:
        payload["embedding"] = learned[0]
    torch.save(payload, out)
    return out


if __name__ == "__main__":
    main()
