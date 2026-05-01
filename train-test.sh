# uv run python src/train-pseudo-token.py \
#   --mode masked-images \
#   --ref-dir experiments/epicphotogasm_zUniversal/persona_008/masked \
#   --placeholder "<pref_008>" --num-tokens 6 \
#   --steps 3600 --lr 5e-4

uv run python src/train-pseudo-token.py \
  --mode masked-images \
  --ref-dir experiments/epicphotogasm_zUniversal/persona_010/masked \
  --placeholder "<pref_010>" --num-tokens 6 \
  --steps 3600 --lr 5e-4
