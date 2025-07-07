#!/usr/bin/env python3
"""
Debug ONNX issue by comparing a simple case
"""

import torch
import audiotools as at
import vampnet_onnx as vampnet
import numpy as np


def debug_onnx_issue():
    """Debug the ONNX implementation issue"""
    
    # Load interfaces
    print("Loading interfaces...")
    interface_onnx = vampnet.interface.Interface.default(use_onnx=True)
    
    # Load test audio
    print("\nLoading test audio...")
    signal = at.AudioSignal("assets/example.wav")
    
    # Encode
    print("Encoding audio...")
    codes = interface_onnx.encode(signal)
    print(f"Codes shape: {codes.shape}")
    
    # Simple test: mask only the middle 10 time steps of all codebooks
    mask = torch.zeros_like(codes)
    start = codes.shape[2] // 2 - 5
    end = codes.shape[2] // 2 + 5
    mask[:, :, start:end] = 1
    
    print(f"\nMask shape: {mask.shape}")
    print(f"Masked region: time steps {start} to {end}")
    print(f"Total masked tokens: {mask.sum().item()}")
    
    # Test with minimal steps
    print("\nGenerating with ONNX (2 steps)...")
    torch.manual_seed(42)
    output = interface_onnx.vamp(
        codes.clone(),
        mask.clone(),
        sampling_steps=2,
        temperature=0.8,
        return_mask=False
    )
    
    # Check preservation
    print("\nChecking token preservation:")
    unmasked_pos = (mask == 0)
    preserved = torch.all(codes[unmasked_pos] == output[unmasked_pos])
    print(f"Unmasked tokens preserved: {preserved}")
    
    # Check specific regions
    before_preserved = torch.all(codes[:, :, :start] == output[:, :, :start])
    after_preserved = torch.all(codes[:, :, end:] == output[:, :, end:])
    masked_changed = torch.any(codes[:, :, start:end] != output[:, :, start:end])
    
    print(f"Before mask preserved: {before_preserved}")
    print(f"After mask preserved: {after_preserved}")
    print(f"Masked region changed: {masked_changed}")
    
    # Count changes
    total_changed = (codes != output).sum().item()
    expected_max = mask.sum().item()
    print(f"\nTotal changed tokens: {total_changed}")
    print(f"Expected max changes: {expected_max}")
    
    # Decode both
    original_signal = interface_onnx.decode(codes)
    output_signal = interface_onnx.decode(output)
    
    original_signal.write("output/debug_original.wav")
    output_signal.write("output/debug_output.wav")
    print("\nSaved debug_original.wav and debug_output.wav")
    
    # Test with no mask (should be identical)
    print("\n=== Testing with no mask ===")
    no_mask = torch.zeros_like(codes)
    torch.manual_seed(42)
    output_no_mask = interface_onnx.vamp(
        codes.clone(),
        no_mask,
        sampling_steps=2,
        temperature=0.8,
        return_mask=False
    )
    
    identical = torch.all(codes == output_no_mask)
    print(f"Output identical to input: {identical}")
    if not identical:
        diff_count = (codes != output_no_mask).sum().item()
        print(f"WARNING: {diff_count} tokens changed with no mask!")
    
    # Test coarse model directly
    print("\n=== Testing coarse model directly ===")
    coarse_codes = codes[:, :4, :]  # First 4 codebooks
    coarse_mask = mask[:, :4, :]
    
    torch.manual_seed(42)
    coarse_output = interface_onnx.coarse_vamp(
        coarse_codes.clone(),
        coarse_mask.clone(),
        sampling_steps=2,
        temperature=0.8,
        return_mask=False
    )
    
    # Check if it returns all 14 codebooks
    print(f"Coarse output shape: {coarse_output.shape}")
    
    # Check preservation in coarse output
    coarse_unmasked = (coarse_mask == 0)
    coarse_preserved = torch.all(coarse_codes[coarse_unmasked] == coarse_output[:, :4, :][coarse_unmasked])
    print(f"Coarse unmasked tokens preserved: {coarse_preserved}")


if __name__ == "__main__":
    import os
    os.makedirs("output", exist_ok=True)
    debug_onnx_issue()