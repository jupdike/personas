#!/bin/bash

# for gpt in 1.0 1.5 2.0 3.0 5.0 7.5; do
#   uv run python src/render-pseudo-token.py \
#     --token-file experiments/epicphotogasm_zUniversal/pref_005.pt \
#     --guidance 7.5 --cfg-pt $gpt --cfg-pt-baseline "woman" \
#     --seed 12345 -n 1
# done

# for gpt in 1.0 1.5 2.0 3.0 5.0 7.5; do
#   uv run python src/render-pseudo-token.py \
#     --token-file experiments/epicphotogasm_zUniversal/pref_005.pt \
#     --guidance 7.5 --cfg-pt $gpt --cfg-pt-baseline "woman" \
#     --init 
#     --seed 12345 -n 1
# done

# uv run python src/render-pseudo-token.py \
#     --token-file experiments/epicphotogasm_zUniversal/pref_005.pt \
#     --prompts-file prompts/test-pt.txt \
#     --init output/IMG_0004.png --strength 0.75 \
#     --guidance 7.5 --cfg-pt 2.2 --cfg-pt-baseline "woman" \
#     --seed 12345 -n 2

# # works to try many input images
# for seed in 0004 0005 0006 0007 0008 0009 0010 0011; do
#     uv run python src/render-pseudo-token.py \
#         --token-file experiments/epicphotogasm_zUniversal/pref_005.pt \
#         --prompts-file prompts/test-pt.txt \
#         --init output/IMG_${seed}.png --strength 0.85 \
#         --guidance 7.5 --cfg-pt 2.2 --cfg-pt-baseline "woman" \
#         --seed 12345 -n 2
# done

# uv run python src/render-pseudo-token.py \
#     --token-file experiments/epicphotogasm_zUniversal/pref_007.pt \
#     --prompts-file prompts/test-pt.txt \
#     --init output/face-crop-IMG_0013.png --strength 0.75 \
#     --guidance 7.5 --cfg-pt 2.2 --cfg-pt-baseline "woman" \
#     --seed 12345 -n 2

# built-in prompts
# uv run python src/render-pseudo-token.py \
#     --token-file experiments/epicphotogasm_zUniversal/pref_007.pt \
#     --guidance 7.5 --cfg-pt 2.2 --cfg-pt-baseline "woman" \
#     --seed 12345 -n 2

# uv run python src/render-pseudo-token.py \
#     --token-file experiments/epicphotogasm_zUniversal/pref_008.pt \
#     --prompts-file prompts/test-pt.txt \
#     --init output/seed-2.png --strength 0.85 \
#     --guidance 6.5 --cfg-pt 2.1 --cfg-pt-baseline "woman" \
#     --seed 123 -n 3

# blend two PTs with slerp
# uv run python src/render-pseudo-token.py \
#     --token-file experiments/epicphotogasm_zUniversal/pref_008.pt \
#     --token-file-b experiments/epicphotogasm_zUniversal/pref_007.pt \
#     --blend-mode slerp \
#     --prompts-file prompts/test-pt.txt \
#     --init output/seed-2.png --strength 0.85 \
#     --guidance 6.5 --cfg-pt 2.1 --cfg-pt-baseline "woman" \
#     --seed 12345 -n 3
# info:
#    p.add_argument("--blend-steps", type=int, default=4,
#                help="Number of intervals in the blend sweep. blend-steps=4 "
#                     "produces 5 images per prompt at alpha = 0, 1/4, 2/4, 3/4, 1.")
#    p.add_argument("--blend-tok-name", default="<blend>",

# uv run python src/render-pseudo-token.py \
#     --token-file experiments/epicphotogasm_zUniversal/pref_007.pt \
#     --scale 0.95 \
#     --prompts-file prompts/test-pt.txt \
#     --init output/seed-2.png --strength 0.85 \
#     --guidance 6.5 --cfg-pt 2.1 --cfg-pt-baseline "woman" \
#     --seed 12345 -n 2

# uv run python src/render-pseudo-token.py \
#     --token-file experiments/epicphotogasm_zUniversal/pref_010.pt \
#     --prompts-file prompts/test-pt.txt \
#     --init output/seed-1.png --strength 0.85 \
#     --guidance 7.5 --cfg-pt 2.4 --cfg-pt-baseline "woman" \
#     --seed 12345 -n 3
