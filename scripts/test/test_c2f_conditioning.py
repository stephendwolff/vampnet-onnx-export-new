#!/usr/bin/env python3
"""
Test that C2F model preserves conditioning codebooks
"""

import torch
import audiotools as at
import vampnet_onnx as vampnet
import numpy as np


def test_c2f_conditioning():
    """Test that C2F preserves the first 4 conditioning codebooks"""
    
    print("Loading ONNX interface...")
    interface = vampnet.interface.Interface.default(use_onnx=True)
    
    # Load test audio
    print("\nLoading test audio...")
    signal = at.AudioSignal("assets/example.wav")
    
    # Encode
    print("Encoding audio...")
    codes = interface.encode(signal)
    print(f"Original codes shape: {codes.shape}")
    
    # Get just the coarse codes
    coarse_codes = codes[:, :4, :]
    print(f"Coarse codes shape: {coarse_codes.shape}")
    
    # Test 1: Direct C2F call with no mask
    print("\n=== Test 1: C2F with no mask (should preserve coarse) ===")
    
    # Create input with 14 codebooks (coarse + zeros for fine)
    c2f_input = torch.zeros_like(codes)
    c2f_input[:, :4, :] = coarse_codes
    
    # No mask means all zeros (nothing to regenerate)
    mask = torch.zeros_like(codes)
    
    c2f_output = interface.coarse_to_fine(
        c2f_input,
        mask=mask,
        sampling_steps=2,
        temperature=0.8,
        return_mask=False
    )
    
    # Check if coarse codes are preserved
    coarse_preserved = torch.all(c2f_output[:, :4, :] == coarse_codes)
    print(f"Coarse codes preserved: {coarse_preserved}")
    
    if not coarse_preserved:
        diff = (c2f_output[:, :4, :] != coarse_codes).sum().item()
        print(f"WARNING: {diff} coarse tokens changed!")
    
    # Test 2: C2F with fine mask only
    print("\n=== Test 2: C2F with fine mask only ===")
    
    # Mask only the fine codebooks
    mask = torch.zeros_like(codes)
    mask[:, 4:, :] = 1  # Mask fine codebooks
    
    c2f_output2 = interface.coarse_to_fine(
        c2f_input,
        mask=mask,
        sampling_steps=4,
        temperature=0.8,
        return_mask=False
    )
    
    # Check if coarse codes are preserved
    coarse_preserved2 = torch.all(c2f_output2[:, :4, :] == coarse_codes)
    print(f"Coarse codes preserved: {coarse_preserved2}")
    
    if not coarse_preserved2:
        diff = (c2f_output2[:, :4, :] != coarse_codes).sum().item()
        print(f"WARNING: {diff} coarse tokens changed!")
    
    # Check if fine codes changed
    fine_changed = torch.any(c2f_output2[:, 4:, :] != c2f_input[:, 4:, :])
    print(f"Fine codes changed: {fine_changed}")
    
    # Test 3: Full vamp pipeline
    print("\n=== Test 3: Full vamp pipeline ===")
    
    # Mask only fine tokens
    mask = torch.zeros_like(codes)
    mask[:, 4:, :] = 1
    
    vamp_output = interface.vamp(
        codes,
        mask,
        sampling_steps=4,
        temperature=0.8,
        return_mask=False
    )
    
    # Check preservation
    coarse_preserved3 = torch.all(vamp_output[:, :4, :] == codes[:, :4, :])
    print(f"Coarse codes preserved in full vamp: {coarse_preserved3}")
    
    if not coarse_preserved3:
        diff = (vamp_output[:, :4, :] != codes[:, :4, :]).sum().item()
        print(f"WARNING: {diff} coarse tokens changed in full pipeline!")
    
    # Decode all versions
    print("\n=== Decoding results ===")
    original_signal = interface.decode(codes)
    c2f_signal1 = interface.decode(c2f_output)
    c2f_signal2 = interface.decode(c2f_output2)
    vamp_signal = interface.decode(vamp_output)
    
    original_signal.write("output/c2f_test_original.wav")
    c2f_signal1.write("output/c2f_test_no_mask.wav")
    c2f_signal2.write("output/c2f_test_fine_mask.wav")
    vamp_signal.write("output/c2f_test_full_vamp.wav")
    
    print("\nSaved outputs:")
    print("- c2f_test_original.wav (original)")
    print("- c2f_test_no_mask.wav (C2F with no mask)")
    print("- c2f_test_fine_mask.wav (C2F with fine mask)")
    print("- c2f_test_full_vamp.wav (full vamp pipeline)")


if __name__ == "__main__":
    import os
    os.makedirs("output", exist_ok=True)
    test_c2f_conditioning()