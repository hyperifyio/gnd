#!/bin/bash
set -e

# Create the embedded directory if it doesn't exist
mkdir -p pkg/bitnet/internal/assets/models/BitNet-b1.58-2B-4T

# Download the model files from Hugging Face
echo "Downloading BitNet model files..."
curl -L "https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf" -o pkg/bitnet/internal/assets/models/BitNet-b1.58-2B-4T/model.bin
curl -L "https://huggingface.co/microsoft/bitnet-b1.58-2B-4T/resolve/main/tokenizer.json" -o pkg/bitnet/internal/assets/models/BitNet-b1.58-2B-4T/tokenizer.json

# Verify the files were downloaded
if [ ! -f pkg/bitnet/internal/assets/models/BitNet-b1.58-2B-4T/model.bin ]; then
    echo "Error: Failed to download model.bin"
    exit 1
fi

if [ ! -f pkg/bitnet/internal/assets/models/BitNet-b1.58-2B-4T/tokenizer.json ]; then
    echo "Error: Failed to download tokenizer.json"
    exit 1
fi

echo "Successfully downloaded BitNet model files" 