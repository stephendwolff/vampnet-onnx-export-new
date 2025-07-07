#!/usr/bin/env python3
"""
Test ONNX generation quality with different mask patterns
"""

import torch
import audiotools as at
import vampnet_onnx as vampnet
import numpy as np


def test_onnx_generation():
    """Test ONNX generation with different mask patterns"""
    
    print("Loading ONNX interface...")
    interface = vampnet.interface.Interface.default(use_onnx=True)
    
    # Load test audio
    print("\nLoading test audio...")
    signal = at.AudioSignal("assets/example.wav")
    signal.write("output/original.wav")
    
    # Encode
    print("Encoding audio...")
    codes = interface.encode(signal)
    print(f"Codes shape: {codes.shape}")
    
    # Test 1: Mask only fine tokens (should sound similar)
    print("\n=== Test 1: Mask only fine tokens ===")
    mask = torch.zeros_like(codes)
    mask[:, 4:, :] = 1  # Mask codebooks 4-13
    
    output1 = interface.vamp(
        codes.clone(),
        mask.clone(),
        sampling_steps=12,
        temperature=0.8,
        return_mask=False
    )
    
    signal1 = interface.decode(output1)
    signal1.write("output/test1_fine_only.wav")
    print("Saved test1_fine_only.wav - should sound very similar to original")
    
    # Test 2: Mask all tokens (complete regeneration)
    print("\n=== Test 2: Mask all tokens ===")
    mask = torch.ones_like(codes)
    
    output2 = interface.vamp(
        codes.clone(),
        mask.clone(),
        sampling_steps=12,
        temperature=0.8,
        return_mask=False
    )
    
    signal2 = interface.decode(output2)
    signal2.write("output/test2_all_masked.wav")
    print("Saved test2_all_masked.wav - should sound very different")
    
    # Test 3: Mask middle portion only
    print("\n=== Test 3: Mask middle portion ===")
    mask = torch.zeros_like(codes)
    start = codes.shape[2] // 3
    end = 2 * codes.shape[2] // 3
    mask[:, :, start:end] = 1
    
    output3 = interface.vamp(
        codes.clone(),
        mask.clone(),
        sampling_steps=12,
        temperature=0.8,
        return_mask=False
    )
    
    signal3 = interface.decode(output3)
    signal3.write("output/test3_middle.wav")
    print("Saved test3_middle.wav - beginning and end should match original")
    
    # Test 4: Progressive masking (coarse to fine)
    print("\n=== Test 4: Progressive masking ===")
    # Keep first 2 codebooks, mask next 2, keep next 4, mask last 6
    mask = torch.zeros_like(codes)
    mask[:, 2:4, :] = 1   # Mask codebooks 2-3
    mask[:, 8:, :] = 1    # Mask codebooks 8-13
    
    output4 = interface.vamp(
        codes.clone(),
        mask.clone(),
        sampling_steps=12,
        temperature=0.8,
        return_mask=False
    )
    
    signal4 = interface.decode(output4)
    signal4.write("output/test4_progressive.wav")
    print("Saved test4_progressive.wav - partial masking across codebooks")
    
    # Test 5: Periodic time masking
    print("\n=== Test 5: Periodic masking ===")
    mask = torch.zeros_like(codes)
    # Mask every 4th time step
    mask[:, :, ::4] = 1
    
    output5 = interface.vamp(
        codes.clone(),
        mask.clone(),
        sampling_steps=12,
        temperature=0.8,
        return_mask=False
    )
    
    signal5 = interface.decode(output5)
    signal5.write("output/test5_periodic.wav")
    print("Saved test5_periodic.wav - periodic time masking")
    
    print("\n=== All tests complete ===")
    print("Listen to the output files to evaluate generation quality.")
    print("\nExpected results:")
    print("- test1_fine_only.wav: Very similar to original (coarse preserved)")
    print("- test2_all_masked.wav: Completely different (full regeneration)")
    print("- test3_middle.wav: Beginning/end match, middle different")
    print("- test4_progressive.wav: Partial similarity based on mask pattern")
    print("- test5_periodic.wav: Rhythmic alterations")


if __name__ == "__main__":
    import os
    os.makedirs("output", exist_ok=True)
    test_onnx_generation()