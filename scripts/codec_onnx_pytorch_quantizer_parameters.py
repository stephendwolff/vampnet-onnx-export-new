# Test if export worked

from pathlib import Path

import numpy as np
import torch
import torch.onnx
from lac.model.lac import LAC as DAC


def get_model_parameters_from_pytorch(pytorch_lac_model):
    """Extract the correct parameters from your PyTorch LAC model"""

    quantizer = pytorch_lac_model.quantizer

    # Get actual parameters
    n_codebooks = len(quantizer.quantizers)
    vocab_size = quantizer.quantizers[0].codebook_size
    latent_dim = quantizer.quantizers[0].codebook_dim

    print(f"Model parameters:")
    print(f"  n_codebooks: {n_codebooks}")
    print(f"  vocab_size: {vocab_size}")
    print(f"  latent_dim: {latent_dim}")
    print(f"  Total latent dimension: {n_codebooks * latent_dim}")

    return n_codebooks, vocab_size, latent_dim


# Get your model's actual parameters

codec_ckpt = str((Path(__file__).parent / "../models/vampnet/codec.pth").resolve())
device = "cpu"
lac_model = DAC.load(codec_ckpt)
lac_model.to(device)
lac_model.eval()

n_codebooks, vocab_size, latent_dim = get_model_parameters_from_pytorch(lac_model)

print(
    f"Model parameters: n_codebooks={n_codebooks}, vocab_size={vocab_size}, latent_dim={latent_dim}"
)

# # Create CodebookEmbedding with correct parameters
# codebook_embedding = CodebookEmbeddingONNX(
#     vocab_size=vocab_size,  # 1024
#     latent_dim=latent_dim,  # 8
#     n_codebooks=n_codebooks,  # 14 (not 4!)
#     emb_dim=256,  # Your desired output embedding size
#     lookup_tables_path="codebook_tables.pth",
#     special_tokens=("MASK", "SEP"),  # This will create [14, 8] special tokens
# )
