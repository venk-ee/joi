#!/usr/bin/env bash
set -euo pipefail

# Download Kokoro-82M ONNX model weights and voice styles
MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"
mkdir -p "${MODELS_DIR}"

KOKORO_MODEL_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
KOKORO_VOICES_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

echo "=== Checking Kokoro TTS Model Files in ${MODELS_DIR} ==="

# 1. Download Model Brain
if [ -f "${MODELS_DIR}/kokoro-v1.0.onnx" ]; then
    echo "✔ kokoro-v1.0.onnx already exists. Skipping."
else
    echo "⬇ Downloading kokoro-v1.0.onnx (~311MB)..."
    curl -L --progress-bar -o "${MODELS_DIR}/kokoro-v1.0.onnx" "${KOKORO_MODEL_URL}"
    echo "✔ Download complete: kokoro-v1.0.onnx"
fi

# 2. Download Voices Styles Pack
if [ -f "${MODELS_DIR}/voices-v1.0.bin" ]; then
    echo "✔ voices-v1.0.bin already exists. Skipping."
else
    echo "⬇ Downloading voices-v1.0.bin (~27MB)..."
    curl -L --progress-bar -o "${MODELS_DIR}/voices-v1.0.bin" "${KOKORO_VOICES_URL}"
    echo "✔ Download complete: voices-v1.0.bin"
fi

echo "=== All Kokoro models are ready! ==="
