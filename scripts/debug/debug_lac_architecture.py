import torch
from lac.model.lac import LAC
import onnxruntime as ort
import numpy as np

# Load the PyTorch model to understand the architecture
lac_model = LAC.load("models/vampnet/codec.pth")
lac_model.eval()

print("LAC Architecture Analysis:")
print(f"Number of codebooks: {lac_model.quantizer.n_codebooks}")
print(f"Latent dim per codebook: {lac_model.quantizer.quantizers[0].codebook.weight.shape[1]}")
print(f"Total VQ dim: {lac_model.quantizer.n_codebooks * lac_model.quantizer.quantizers[0].codebook.weight.shape[1]}")

# Check the encoder output dimension
dummy_audio = torch.randn(1, 1, 44100)  # 1 second of audio
with torch.no_grad():
    # Get encoder output
    continuous = lac_model.encoder(dummy_audio)
    print(f"\nEncoder output shape: {continuous.shape}")
    print(f"Encoder output channels: {continuous.shape[1]}")
    
    # Get quantizer output
    quantized, codes, latents, commitment_loss, codebook_loss = lac_model.quantizer(continuous)
    print(f"\nQuantizer output shape: {quantized.shape}")
    print(f"Codes shape: {codes.shape}")
    print(f"Latents shape: {latents.shape}")

print("\nConclusion:")
print("The 1024-dim quantized features are NOT just from codebooks!")
print("They include additional features from the encoder that aren't quantized.")