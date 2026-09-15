#!/bin/bash
# Download all models for the decision support system.
# Suitable for Mac Studio Ultra M3 (96GB) or RTX 6000 Pro Blackwell (96GB).
# Requires Ollama >= 0.20 for Gemma 4. Not needed with vLLM/mlx-lm; see README.

set -e

models=(
    "gemma4:26b"
    "gemma4:31b"
    "gemma4:12b"
    "qwen2.5:72b"
    "llama3.3:70b"
    "mistral-small:24b"
)

for model in "${models[@]}"; do
    echo "=== Pulling $model ==="
    ollama pull "$model"
    echo ""
done

echo "=== Downloading embedding and re-ranking models ==="
uv run worker.py --download

echo "All models downloaded."
