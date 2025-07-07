#!/usr/bin/env python3
"""
Test script for using the exported VampNet coarse ONNX models.
"""

import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path


def test_coarse_inference():
    """Test inference with the exported ONNX models"""
    
    # Load ONNX models
    embed_path = Path(__file__).parent.parent / "vampnet_coarse_embeddings.onnx"
    trans_path = Path(__file__).parent.parent / "vampnet_coarse_transformer.onnx"
    
    print("Loading ONNX models...")
    embed_session = ort.InferenceSession(str(embed_path))
    trans_session = ort.InferenceSession(str(trans_path))
    
    # Create test input - discrete codes
    batch_size = 1
    n_codebooks = 4  # Coarse model uses 4 codebooks
    time_steps = 20
    vocab_size = 1024
    
    # Random codes for testing
    codes = np.random.randint(0, vocab_size, (batch_size, n_codebooks, time_steps), dtype=np.int64)
    print(f"Input codes shape: {codes.shape}")
    
    # Step 1: Convert codes to latents
    latents = embed_session.run(None, {"codes": codes})[0]
    print(f"Latents shape: {latents.shape}")
    
    # Step 2: Process through transformer
    logits = trans_session.run(None, {"latents": latents})[0]
    print(f"Logits shape: {logits.shape}")
    
    # The logits can be reshaped to separate codebooks
    logits_reshaped = logits.reshape(batch_size, n_codebooks, vocab_size, time_steps)
    print(f"Logits reshaped: {logits_reshaped.shape}")
    
    # In practice, you would:
    # 1. Sample from these logits to generate new codes
    # 2. Use a mask to control which positions to generate
    # 3. Iteratively refine the generation
    
    print("\nTo use these models in a generation pipeline:")
    print("1. Start with masked tokens or partial codes")
    print("2. Run through embeddings → transformer → logits")
    print("3. Sample from logits to get new codes")
    print("4. Repeat for multiple steps with decreasing mask ratio")
    
    return logits


def demonstrate_masking():
    """Demonstrate how masking works in VampNet"""
    
    # The mask token is typically vocab_size (1024)
    mask_token = 1024
    
    # Create a sequence with some masked positions
    batch_size = 1
    n_codebooks = 4
    time_steps = 10
    
    # Start with all mask tokens
    codes = np.full((batch_size, n_codebooks, time_steps), mask_token, dtype=np.int64)
    
    # Unmask some positions (e.g., condition on first and last frame)
    codes[:, :, 0] = np.random.randint(0, 1024, (batch_size, n_codebooks))
    codes[:, :, -1] = np.random.randint(0, 1024, (batch_size, n_codebooks))
    
    print("Example masked codes (1024 = mask token):")
    print(codes[0, 0, :])  # Show first codebook
    
    return codes


if __name__ == "__main__":
    print("Testing VampNet coarse ONNX models...")
    print("-" * 50)
    
    # Test basic inference
    test_coarse_inference()
    
    print("\n" + "-" * 50)
    print("Masking example:")
    demonstrate_masking()