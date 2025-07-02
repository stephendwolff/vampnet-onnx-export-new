# Test if export worked

from pathlib import Path

import numpy as np
import torch
import torch.onnx
from lac.model.lac import LAC as DAC


def extract_codebook_lookup_tables(pytorch_lac_model, save_path):
    """Extract codebook lookup tables for VampNet"""

    lookup_tables = []

    # Extract weights from each quantizer
    for i, quantizer in enumerate(pytorch_lac_model.quantizer.quantizers):
        codebook_weight = (
            quantizer.codebook.weight.data.clone()
        )  # Shape: (codebook_size, codebook_dim)
        lookup_tables.append(codebook_weight)
        print(f"Codebook {i}: {codebook_weight.shape}")

    # Save all lookup tables
    torch.save(lookup_tables, save_path)
    print(f"✓ Saved {len(lookup_tables)} lookup tables to {save_path}")

    return lookup_tables


codec_ckpt = str((Path(__file__).parent / "../models/vampnet/codec.pth").resolve())
device = "cpu"
lac_model = DAC.load(codec_ckpt)
lac_model.to(device)
lac_model.eval()

# Extract the lookup tables
lookup_tables = extract_codebook_lookup_tables(lac_model, "lac_codebook_tables.pth")
