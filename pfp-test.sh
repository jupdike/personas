#!/bin/bash
set -euo pipefail
BASE_DIR=experiments/epicphotogasm_zUniversal/persona_05_06

mkdir -p "${BASE_DIR}/masked"

for ref in "${BASE_DIR}/refs/"*.png; do
  STEM=$(basename "$ref" .png)
  MASK="${BASE_DIR}/resnet34/${STEM}_raw.png"
  if [[ ! -f "$MASK" ]]; then
    echo "skip ${STEM}: no mask at ${MASK}"
    continue
  fi
  echo "Combining ${STEM}.png with face parse mask..."
  uv run python src/parse-face-parse.py \
    "$ref" \
    "$MASK" \
    "${BASE_DIR}/masked/${STEM}.png"
done
