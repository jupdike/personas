#!/bin/bash
set -euo pipefail
BASE_DIR=experiments/epicphotogasm_zUniversal/persona_010
ROOT_DIR=~/Documents/dev/jfu/personas

mkdir -p "${BASE_DIR}/masked"
# resnet34/ will be added by face-parsing/inference.py, for output folder
face-parse.sh --input "${ROOT_DIR}/${BASE_DIR}/refs" --output "${ROOT_DIR}/${BASE_DIR}"

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
