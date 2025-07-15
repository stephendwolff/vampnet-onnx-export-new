#!/bin/bash

set -e

# Ensure pyenv and Python 3.11 are available
if ! command -v pyenv &> /dev/null; then
  echo "pyenv not found. Please install pyenv first."
  exit 1
fi

pyenv install -s 3.11
pyenv local 3.11

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Check for models
if [ ! -f models/vampnet/c2f.pth ] || [ ! -f models/vampnet/coarse.pth ] || [ ! -f models/vampnet/codec.pth ] || [ ! -f models/wavebeat.pth ]; then
  echo "Model files not found in /models. Please download them as per the README."
  exit 1
fi

# Run export scripts
python scripts/codec_onnx_export.py
python scripts/export_from_codes_with_proj.py
python scripts/codec_onnx_extract_codebooks.py
python scripts/codec_onnx_export_quantizer.py
python scripts/coarse_onnx_export.py
python scripts/c2f_onnx_export.py

echo "ONNX export complete."
