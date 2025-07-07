#!/usr/bin/env python3
"""
Test masking behavior in VampNet ONNX to ensure unmasked tokens are preserved
"""

import torch
import audiotools as at
import vampnet_onnx as vampnet
import numpy as np


def test_masking_preservation():
    """Test that unmasked tokens remain unchanged during generation"""
    
    print("Loading ONNX interface...")
    interface = vampnet.interface.Interface.default(use_onnx=True)
    
    # Load test audio
    print("\nLoading test audio...")
    signal = at.AudioSignal("assets/example.wav")
    
    # Encode to codes
    print("Encoding audio to codes...")
    codes = interface.encode(signal)
    print(f"Original codes shape: {codes.shape}")
    
    # Test 1: Keep all coarse tokens, mask all fine tokens
    print("\n=== Test 1: Keep coarse, mask fine tokens ===")
    mask = torch.zeros_like(codes)
    mask[:, 4:, :] = 1  # Mask codebooks 4-13 (fine tokens)
    
    # Count how many tokens we're keeping vs masking
    total_tokens = codes.numel()
    masked_tokens = mask.sum().item()
    kept_tokens = total_tokens - masked_tokens
    print(f"Total tokens: {total_tokens}")
    print(f"Kept tokens: {kept_tokens} ({kept_tokens/total_tokens*100:.1f}%)")
    print(f"Masked tokens: {masked_tokens} ({masked_tokens/total_tokens*100:.1f}%)")
    
    # Generate with mask
    output_codes = interface.vamp(
        codes,
        mask,
        sampling_steps=8,
        temperature=0.8,
        return_mask=False
    )
    
    # Check if unmasked tokens were preserved
    unmasked_positions = (mask == 0)
    preserved = torch.all(codes[unmasked_positions] == output_codes[unmasked_positions])
    print(f"\nCoarse tokens preserved: {preserved}")
    
    # Check statistics
    changed_tokens = (codes != output_codes).sum().item()
    print(f"Changed tokens: {changed_tokens}")
    print(f"Expected changed tokens: <= {masked_tokens}")
    
    # Decode both
    original_signal = interface.decode(codes)
    output_signal = interface.decode(output_codes)
    
    # Save outputs
    original_signal.write("output/test1_original.wav")
    output_signal.write("output/test1_coarse_kept.wav")
    print("Saved: test1_original.wav and test1_coarse_kept.wav")
    
    # Test 2: Periodic masking pattern
    print("\n=== Test 2: Periodic masking pattern ===")
    mask = torch.zeros_like(codes)
    # Mask every other time step
    mask[:, :, ::2] = 1
    
    masked_tokens = mask.sum().item()
    kept_tokens = total_tokens - masked_tokens
    print(f"Kept tokens: {kept_tokens} ({kept_tokens/total_tokens*100:.1f}%)")
    print(f"Masked tokens: {masked_tokens} ({masked_tokens/total_tokens*100:.1f}%)")
    
    output_codes = interface.vamp(
        codes,
        mask,
        sampling_steps=8,
        temperature=0.8,
        return_mask=False
    )
    
    # Check preservation
    unmasked_positions = (mask == 0)
    preserved = torch.all(codes[unmasked_positions] == output_codes[unmasked_positions])
    print(f"\nAlternating tokens preserved: {preserved}")
    
    output_signal = interface.decode(output_codes)
    output_signal.write("output/test2_periodic_mask.wav")
    print("Saved: test2_periodic_mask.wav")
    
    # Test 3: Small masked region
    print("\n=== Test 3: Small masked region ===")
    mask = torch.zeros_like(codes)
    # Mask a small region in the middle
    start = codes.shape[2] // 3
    end = 2 * codes.shape[2] // 3
    mask[:, :, start:end] = 1
    
    masked_tokens = mask.sum().item()
    kept_tokens = total_tokens - masked_tokens
    print(f"Kept tokens: {kept_tokens} ({kept_tokens/total_tokens*100:.1f}%)")
    print(f"Masked tokens: {masked_tokens} ({masked_tokens/total_tokens*100:.1f}%)")
    
    output_codes = interface.vamp(
        codes,
        mask,
        sampling_steps=8,
        temperature=0.8,
        return_mask=False
    )
    
    # Check preservation  
    unmasked_positions = (mask == 0)
    preserved = torch.all(codes[unmasked_positions] == output_codes[unmasked_positions])
    print(f"\nUnmasked regions preserved: {preserved}")
    
    # Check specific regions
    before_preserved = torch.all(codes[:, :, :start] == output_codes[:, :, :start])
    after_preserved = torch.all(codes[:, :, end:] == output_codes[:, :, end:])
    print(f"Region before mask preserved: {before_preserved}")
    print(f"Region after mask preserved: {after_preserved}")
    
    output_signal = interface.decode(output_codes)
    output_signal.write("output/test3_middle_masked.wav")
    print("Saved: test3_middle_masked.wav")
    
    # Test 4: No masking (should return identical)
    print("\n=== Test 4: No masking (sanity check) ===")
    mask = torch.zeros_like(codes)  # No masking
    
    output_codes = interface.vamp(
        codes,
        mask,
        sampling_steps=8,
        temperature=0.8,
        return_mask=False
    )
    
    identical = torch.all(codes == output_codes)
    print(f"Output identical to input: {identical}")
    
    if not identical:
        changed = (codes != output_codes).sum().item()
        print(f"WARNING: {changed} tokens changed when none should have!")
    
    print("\n=== All tests complete ===")
    print("Check the output/ directory for generated audio files")


if __name__ == "__main__":
    import os
    os.makedirs("output", exist_ok=True)
    test_masking_preservation()