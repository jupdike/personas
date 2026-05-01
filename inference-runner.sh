uv run python src/inference-test.py \
    --template-file prompts/templates.txt \
    --strength 0.85 \
    --guidance 7.5 --steps 25\
    --seed 54321 -n 3
