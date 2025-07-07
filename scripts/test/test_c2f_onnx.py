#!/usr/bin/env python3
"""
Test script for using the exported VampNet c2f ONNX models.
"""

import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path


def test_c2f_inference():
    """Test inference with the exported c2f ONNX models"""
    
    # Load ONNX models
    embed_path = Path(__file__).parent.parent / "vampnet_c2f_embeddings.onnx"
    trans_path = Path(__file__).parent.parent / "vampnet_c2f_transformer.onnx"
    
    print("Loading ONNX models...")
    embed_session = ort.InferenceSession(str(embed_path))
    trans_session = ort.InferenceSession(str(trans_path))
    
    # Create test input - all 14 codebooks
    batch_size = 1
    n_total_codebooks = 14
    n_conditioning_codebooks = 4
    n_predict_codebooks = 10
    time_steps = 20
    vocab_size = 1024
    mask_token = 1024
    
    # Create codes with conditioning (first 4) filled and rest masked
    codes = np.full((batch_size, n_total_codebooks, time_steps), mask_token, dtype=np.int64)
    
    # Fill conditioning codebooks with random values (simulating coarse output)
    codes[:, :n_conditioning_codebooks, :] = np.random.randint(
        0, vocab_size, (batch_size, n_conditioning_codebooks, time_steps)
    )
    
    print(f"Input codes shape: {codes.shape}")
    print(f"Conditioning codebooks (0-3): filled with values")
    print(f"Fine codebooks (4-13): filled with mask token ({mask_token})")
    
    # Step 1: Convert codes to latents
    latents = embed_session.run(None, {"codes": codes})[0]
    print(f"\nLatents shape: {latents.shape}")
    
    # Step 2: Process through transformer
    logits = trans_session.run(None, {"latents": latents})[0]
    print(f"Logits shape: {logits.shape}")
    
    # The logits are for the non-conditioning codebooks only
    logits_reshaped = logits.reshape(batch_size, n_predict_codebooks, vocab_size, time_steps)
    print(f"Logits reshaped: {logits_reshaped.shape}")
    
    print("\nThe c2f model:")
    print(f"- Input: all {n_total_codebooks} codebooks")
    print(f"- Conditioning: first {n_conditioning_codebooks} codebooks (from coarse model)")
    print(f"- Output: logits for remaining {n_predict_codebooks} codebooks")
    
    return logits


def demonstrate_full_pipeline():
    """Demonstrate the full coarse + c2f pipeline"""
    
    print("\nFull VampNet Pipeline:")
    print("=" * 50)
    
    print("\n1. COARSE STAGE:")
    print("   - Input: masked tokens or partial codes")
    print("   - Output: first 4 codebooks (coarse representation)")
    print("   - Uses iterative masked generation")
    
    print("\n2. COARSE-TO-FINE STAGE:")
    print("   - Input: all 14 codebooks")
    print("     - Codebooks 0-3: from coarse stage (conditioning)")
    print("     - Codebooks 4-13: mask tokens")
    print("   - Output: logits for codebooks 4-13")
    print("   - Can also use iterative generation for better quality")
    
    print("\n3. DECODE TO AUDIO:")
    print("   - Input: all 14 codebooks")
    print("   - Use lac_from_codes.onnx to convert codes to features")
    print("   - Use lac_decoder.onnx to convert features to audio")
    
    # Show example dimensions
    batch = 1
    time = 100  # ~1.7 seconds at 44.1kHz with hop_length 768
    
    print(f"\nExample dimensions for {time} time steps:")
    print(f"  Coarse input: ({batch}, 4, {time})")
    print(f"  Coarse output: ({batch}, 4, {time})")
    print(f"  C2F input: ({batch}, 14, {time})")
    print(f"  C2F output: ({batch}, 10, 1024, {time})")
    print(f"  Final codes: ({batch}, 14, {time})")
    print(f"  Audio output: ({batch}, 1, {time * 768})")  # hop_length = 768


if __name__ == "__main__":
    print("Testing VampNet c2f ONNX models...")
    print("-" * 50)
    
    # Test c2f inference
    test_c2f_inference()
    
    # Show full pipeline
    demonstrate_full_pipeline()