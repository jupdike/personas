#!/bin/bash

for gpt in 1.0 1.5 2.0 3.0 5.0 7.5; do
  uv run python src/render-pseudo-token.py \
    --token-file experiments/epicphotogasm_zUniversal/pref_005.pt \
    --guidance 7.5 --cfg-pt $gpt --cfg-pt-baseline "woman" \
    --seed 12345 -n 1
done
