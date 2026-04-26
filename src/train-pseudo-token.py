"""Distill a single source prompt into a 768-dim CLIP pseudo-token.

Reads:   {experiment_path}/persona.txt
Writes:  {experiment_path}/{placeholder}.pt

See PLAN.md for the rationale; this file implements the U-Net prediction
distillation loop described there.
"""
import argparse
import hashlib
import json
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline

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


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--placeholder", default="<pref_001>",
                   help="Placeholder token to add to the tokenizer")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init-mode", choices=["persona", "words"], default="persona",
                   help="persona: initialize from the mean of the persona's own token "
                        "embeddings (puts the placeholder in the right region from step "
                        "0). words: legacy behavior, mean of --init-words.")
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


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = "mps"
    dtype = torch.float32 if args.dtype == "fp32" else torch.float16

    persona_path = f"{experiment_path}/persona.txt"
    persona = open(persona_path).read().strip()
    print(f"<> Model: {model_name}")
    print(f"<> Experiment path: {experiment_path}")
    print(f"<> Persona: {persona}")
    print(f"<> Placeholder: {args.placeholder}")
    print(f"<> Steps: {args.steps}  lr: {args.lr}  dtype: {args.dtype}  "
          f"init: {args.init_mode}  timestep-bias: {args.timestep_bias}")

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

    # We don't need the VAE for distillation training.
    pipe.vae = None

    # Add the placeholder token and resize the embedding table.
    num_added = tokenizer.add_tokens([args.placeholder])
    if num_added == 0:
        raise SystemExit(
            f"Token {args.placeholder!r} already exists in the tokenizer; "
            "pick a different placeholder."
        )
    text_encoder.resize_token_embeddings(len(tokenizer))
    placeholder_id = tokenizer.convert_tokens_to_ids(args.placeholder)
    print(f"<> Placeholder token id: {placeholder_id} "
          f"(vocab size now {len(tokenizer)})")

    embedding_layer = text_encoder.get_input_embeddings()
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    if args.init_mode == "persona":
        # Tokenize the persona string and average its content tokens (no specials).
        persona_ids = tokenizer(persona, add_special_tokens=False).input_ids
        persona_ids = [i for i in persona_ids
                       if i not in (bos_id, eos_id, pad_id, placeholder_id)]
        if not persona_ids:
            raise SystemExit("Persona tokenized to zero content tokens; cannot init.")
        init_ids = persona_ids
        init_label = f"persona ({len(init_ids)} tokens)"
    else:
        words = [w.strip() for w in args.init_words.split(",") if w.strip()]
        init_ids = tokenizer.convert_tokens_to_ids(words)
        init_label = f"words {words}"
    with torch.no_grad():
        init_vec = embedding_layer.weight[init_ids].float().mean(0)
        embedding_layer.weight[placeholder_id] = init_vec.to(embedding_layer.weight.dtype)
    print(f"<> Initialized from mean of {init_label} -> norm {init_vec.norm().item():.3f}")

    # Freeze everything. The embedding weight needs requires_grad=True so autograd
    # produces a gradient for our row; we then mask all other rows to zero.
    for p in text_encoder.parameters():
        p.requires_grad_(False)
    for p in unet.parameters():
        p.requires_grad_(False)
    embedding_layer.weight.requires_grad_(True)

    # fp32 master copy of just the placeholder row. The frozen pipeline can stay in
    # fp16 (fast on MPS, low memory), but the optimizer state and weight update must
    # be in fp32 to avoid NaN from underflow on tiny noise-prediction MSE gradients.
    master = nn.Parameter(
        embedding_layer.weight[placeholder_id].detach().float().clone()
    )
    optimizer = torch.optim.AdamW([master], lr=args.lr)

    num_train_timesteps = pipe.scheduler.config.num_train_timesteps

    losses = []
    t0 = time.time()
    last_log = t0
    print(f"<> Starting training for {args.steps} steps ...")
    for step in range(args.steps):
        # Push the current fp32 master row into the (possibly-fp16) embedding table.
        with torch.no_grad():
            embedding_layer.weight.data[placeholder_id] = master.data.to(
                embedding_layer.weight.dtype
            )

        template = random.choice(SCENE_TEMPLATES)
        target_prompt = template.format(s=persona)
        pred_prompt = template.format(s=args.placeholder)

        target_ids = tokenizer(
            target_prompt, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)
        pred_ids = tokenizer(
            pred_prompt, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)

        z = torch.randn(1, 4, 64, 64, device=device, dtype=dtype)
        if args.timestep_bias == "identity" and random.random() < 0.7:
            # Identity-shaping band: text most influences the image around mid-noise.
            t = torch.randint(300, 900, (1,), device=device).long()
        else:
            t = torch.randint(0, num_train_timesteps, (1,), device=device).long()

        with torch.no_grad():
            target_hs = text_encoder(target_ids)[0]
            eps_target = unet(z, t, encoder_hidden_states=target_hs).sample

        pred_hs = text_encoder(pred_ids)[0]
        eps_pred = unet(z, t, encoder_hidden_states=pred_hs).sample

        # Compute the loss in fp32 — the noise-prediction MSE is small (~5e-4) and
        # underflows to zero in fp16, which produced NaNs in earlier runs.
        loss = F.mse_loss(eps_pred.float(), eps_target.float())

        embedding_layer.weight.grad = None
        master.grad = None
        loss.backward()

        # Pull the gradient for our row out of the (fp16) embedding table, cast to
        # fp32, and apply the AdamW step on the master copy.
        if embedding_layer.weight.grad is not None:
            master.grad = (
                embedding_layer.weight.grad[placeholder_id].detach().float().clone()
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
            norm = master.detach().norm().item()
            print(f"step {step:5d}  loss={loss.item():.6f}  avg{args.log_every}={avg:.6f}  "
                  f"||emb||={norm:.3f}  ({sps:.2f} step/s)")

        if args.save_every and step > 0 and step % args.save_every == 0:
            save_token(args, persona, master.detach(), step)

    elapsed = time.time() - t0
    print(f"<> Done in {elapsed:.1f}s ({args.steps / elapsed:.2f} step/s)")

    out_path = save_token(args, persona, master.detach(), args.steps)
    print(f"<> Saved final pseudo-token to {out_path}")

    # Sidecar JSON for human inspection (the .pt itself is the source of truth).
    meta_path = out_path.replace(".pt", ".json")
    with open(meta_path, "w") as f:
        json.dump({
            "token": args.placeholder,
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
        }, f, indent=2)
    print(f"<> Metadata sidecar at {meta_path}")


def save_token(args, persona: str, embedding: torch.Tensor, step: int) -> str:
    name = args.placeholder.strip("<>").replace(":", "_").replace("/", "_")
    if args.out:
        out = args.out
    else:
        out = f"{experiment_path}/{name}.pt"
    learned = embedding.detach().float().cpu().clone()
    torch.save({
        "token": args.placeholder,
        "embedding": learned,
        "checkpoint_hash": file_sha256(model_path),
        "checkpoint_name": model_name,
        "source_prompt": persona,
        "num_steps": step,
        "lr": args.lr,
    }, out)
    return out


if __name__ == "__main__":
    main()
