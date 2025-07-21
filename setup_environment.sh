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
