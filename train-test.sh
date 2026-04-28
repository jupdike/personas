uv run python src/train-pseudo-token.py \
  --mode masked-images \
  --ref-dir experiments/epicphotogasm_zUniversal/persona_05_06/masked \
  --placeholder "<pref_007>" --num-tokens 6 \
  --steps 3600 --lr 5e-4
