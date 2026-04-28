# Textual Inversion via Prompt Distillation — Experiment Plan

## Goal

Train a single pseudo-token (a learnable 768-dim input embedding for CLIP) that reproduces the behavior of a known source prompt when rendered through Stable Diffusion 1.5. The pseudo-token should work as a drop-in "word" in arbitrary future prompts, e.g. `"<pref_42>, at the beach, soft lighting"`, producing the persona encoded by the source prompt in arbitrary new scenes.

This avoids the standard image-based textual inversion pipeline entirely. Instead of collecting or generating reference images, we distill the U-Net's response to the original prompt directly into the pseudo-token.

## Why this approach fits the use case

- **The source prompt is already ground truth.** The persona exists because some text prompt produces it through CLIP+U-Net. Distilling from the prompt skips the noisy intermediate step of generating and re-encoding images.
- **The target persona lives on CLIP's manifold.** Base CLIP was trained on natural-language captions; a prompt like "a woman with green eyes and dark hair" is already well-represented in its input-embedding space. TI is being used for projection, not concept acquisition.
- **No image dataset needed.** Training is fully synthetic: random noise latents, random timesteps, deterministic prompt pairs.
- **Cheap iteration.** Each training step is one CLIP forward + two U-Net forwards + backprop on 768 parameters. Under a minute per persona on Apple Silicon MPS for a basic run.

The output artifact is a ~3 KB `.safetensors` file containing one 768-float vector, usable as a drop-in "word" in any prompt on the same SD checkpoint the pseudo-token was trained against.

## Technical recap

### What textual inversion is

Add one new row to CLIP's token embedding table. That row is the only trainable parameter; all other weights in CLIP, U-Net, and VAE are frozen.

### Shapes at a glance

| Tensor | Shape | Trainable? |
|---|---|---|
| CLIP token embedding table | `[~49408, 768]` | one row only |
| **The trainable pseudo-token** | `[768]` | **yes** |
| Token ids (tokenized prompt) | `[1, 77]` | no |
| Input embeddings (post-lookup) | `[1, 77, 768]` | no (except 1 slot) |
| CLIP hidden states (text transformer output) | `[1, 77, 768]` | no |
| Latent `z` | `[1, 4, 64, 64]` | no (random) |
| Timestep `t` | scalar | no (random) |
| U-Net noise prediction | `[1, 4, 64, 64]` | no |
| Loss | scalar | — |

Total trainable parameters: **768 floats** (~3 KB).

### Why 768 floats is enough

The source prompt is typically ~100–150 bytes of text. CLIP+transformer is a large expansion machine that turns those bytes into a 237 KB contextualized `[1, 77, 768]` tensor. Storing the persona as a 3 KB pseudo-token is not compression — it is roughly 30× the information density of the original prompt. The transformer does the heavy lifting of expanding the pseudo-token into a contextualized sequence at inference time.

For personas that already live on CLIP's manifold (descriptions in natural language), a single token is usually sufficient. For harder cases, multi-token TI (2–4 new rows, ~6–12 KB) is a straightforward upgrade.

## Core training loop: U-Net prediction distillation

The principle: the pseudo-token should make the U-Net produce the same noise prediction as the full source prompt would, on arbitrary random latents at arbitrary timesteps. Variety across scenes is introduced by varying the prompt wrapper around both the source description and the pseudo-token.

### Gradient masking

When you set `embedding_layer.weight.requires_grad_(True)`, *all* rows receive gradients. That will drift the entire vocabulary and ruin the model. Either:

1. After `loss.backward()`, zero out the gradient on every row except `placeholder_id` (as in the pseudocode), or
2. Use a custom `Parameter` that only holds the one row and gets manually swapped into the table before each forward.

Option 1 is simpler; option 2 is slightly more memory-efficient. Both work.

## Diversity: the core trick

The pseudo-token must encode *only the persona*, not any particular scene. This is achieved by pairing target and prediction with the **same scene wrapper** at each step. The scene is constant between target and pred, so it cancels out of the loss — the only gradient signal on the pseudo-token is about reproducing the subject.

### Strategies

- **30–50 varied scene templates.** Generate with an LLM in one shot: `"give me 40 short photography-style scene/setting descriptions, each used as a comma-separated suffix after a subject named {s}. Vary location, lighting, time of day, style, and composition."`
- **Include bare templates** like `"{s}"` and `"a photo of {s}"` so the pseudo-token also works in minimal prompts.
- **Vary the persona phrasing** lightly in target prompts (`"a woman with green eyes"` vs. `"a green-eyed woman"`) to prevent the pseudo-token from locking onto one exact phrasing. Optional, subtle regularization.
- **Keep the pseudo-token position consistent** across templates — always substitute into the same `{s}` slot. This is handled automatically by using string templates.

## Practical hyperparameters

| Parameter | Suggested value | Notes |
|---|---|---|
| Steps | 1500–3000 | Watch loss plateau; overfit risk is low given regularization from scene variety |
| Learning rate | 5e-4 to 1e-3 | AdamW, no schedule needed for this size |
| Batch size | 1 | Single-step loop; gradient accumulation optional |
| Optimizer | AdamW | Standard; no tricks needed |
| Initialization | Mean of related word embeddings | e.g. average of `"woman"`, `"portrait"` input embeddings for a woman persona |
| Timestep sampling | Uniform over `[0, 1000)` | Optional: oversample `[300, 900]` where identity is most shaped |
| Precision | fp32 on MPS | fp16 is unreliable on Apple Silicon; keep fp32 |
| Scene templates | 30–50 | LLM-generated once, reused |

Expected wall-clock on M1 Pro with 16 GB RAM at 512×512 / SD 1.5:
- ~0.3–0.8 sec per step
- ~10–30 minutes total per persona

## Validation procedure

After training, load the learned `.pt` and test:

1. **Solo fidelity:** prompt `"a photo of <pref_42>"`. Should produce the persona clearly, matching the look from the source prompt.
2. **Novel scene composition:** prompts not in the training template bank, e.g. `"<pref_42>, medieval armor, castle courtyard"` or `"<pref_42>, reading a book, library"`. Should preserve identity while fully rendering the new scene.
3. **Modifier compatibility:** style prompts like `"<pref_42>, in the style of Vermeer"` or `"<pref_42>, pencil sketch"`. The persona should bend stylistically without losing identity.
4. **Token-budget check:** write a long, scene-rich prompt `"<pref_42>, sitting at a cafe, autumn leaves falling, golden afternoon light, holding a book, shallow depth of field, film grain"`. Should render all elements without cutting identity short. This validates the "freed-tokens" win.

If solo fidelity is good but novel scenes degrade identity, add more diverse scene templates and train more steps. If solo fidelity is poor, consider multi-token TI (train 2–4 new rows instead of 1).

## Deliverables

- `pref_42.pt` (or `.safetensors`): single-row embedding, ~3 KB.
- Metadata in the file: placeholder string, SD checkpoint hash, training config snapshot. Useful for matching pseudo-tokens to the right U-Net at inference time.

```python
torch.save({
    "token": placeholder,
    "embedding": learned,
    "checkpoint_hash": "<sha256 of the .safetensors used>",
    "source_prompt": persona,
    "num_steps": num_steps,
    "lr": 5e-4,
}, "pref_42.pt")
```

## Environment setup

```bash
pip install diffusers transformers accelerate safetensors torch
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

SD 1.5 checkpoint recommendations (pick a fine-tune that produces good humans):
- Realistic Vision v5.1 or v6.0
- epiCRealism
- DreamShaper v8

Download `.safetensors` from Civitai or HuggingFace. Load via `StableDiffusionPipeline.from_single_file(...)`. Train the pseudo-token against the same checkpoint you intend to use at inference — pseudo-tokens are coupled to the U-Net weights they were trained against.

## Optional enhancements (defer until baseline works)

- **Multi-token TI.** Train 2–4 new rows instead of 1 for better fidelity on detailed personas. Change `add_tokens` to add multiple placeholders and substitute a multi-word run in the template.
- **CFG-aware distillation.** Compute both conditional and unconditional U-Net predictions on target and pred; distill at the CFG-combined output. Helps pseudo-token behave correctly at high guidance scales.
- **Pooled CLIP warm start.** Before the expensive U-Net distillation loop, run a fast pooled-CLIP-output MSE loop for ~500 steps to get the pseudo-token into a sensible region of embedding space. Can cut total training time.
- **Timestep emphasis.** Sample `t` from a distribution weighted toward 300–900 (where identity is established) rather than uniform.
- **Scene-set augmentation via LLM loop.** Periodically expand the template bank with fresh LLM-generated scenes to reduce memorization risk.

## Next step after this works

Once single-persona training is validated, the path to the larger "pref=int64" system is:

1. Build a corpus of 256–1024 pseudo-tokens, one per curated source prompt, using this training loop in batch.
2. Collect the resulting 768-vectors into a point cloud.
3. Train a small generator `G(z) → [768]` (MLP / small VAE / normalizing flow) on that point cloud, with latent `z` of dim 16–32.
4. At inference: `int64 → PRNG → z → G(z) → pseudo-token row` spliced into the embedding table.

Each stage is self-contained and produces a shippable artifact. No need to commit to the full generator project before the single-persona baseline is solid.
