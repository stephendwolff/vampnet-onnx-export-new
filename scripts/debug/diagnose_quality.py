#!/usr/bin/env python3
"""
Diagnose audio quality issues with different parameters
"""

import torch
import audiotools as at
import vampnet_onnx as vampnet
import numpy as np


def diagnose_quality():
    """Test different parameters to diagnose quality issues"""
    
    print("Loading ONNX interface...")
    interface = vampnet.interface.Interface.default(use_onnx=True)
    
    # Load test audio
    print("\nLoading test audio...")
    signal = at.AudioSignal("assets/example.wav")
    signal.write("output/diag_original.wav")
    
    # Encode
    print("Encoding audio...")
    codes = interface.encode(signal)
    print(f"Codes shape: {codes.shape}")
    
    # Test different scenarios
    tests = [
        {
            "name": "direct_decode",
            "description": "Direct decode without VampNet",
            "mask": torch.zeros_like(codes),  # No masking
            "sampling_steps": 0,  # Skip VampNet
            "temperature": 1.0,
        },
        {
            "name": "minimal_mask",
            "description": "Mask only 1 time step",
            "mask": torch.zeros_like(codes),
            "sampling_steps": 4,
            "temperature": 0.8,
        },
        {
            "name": "low_temp",
            "description": "Fine tokens with low temperature",
            "mask": torch.zeros_like(codes),
            "sampling_steps": 8,
            "temperature": 0.5,
        },
        {
            "name": "high_temp",
            "description": "Fine tokens with high temperature",
            "mask": torch.zeros_like(codes),
            "sampling_steps": 8,
            "temperature": 1.2,
        },
        {
            "name": "more_steps",
            "description": "Fine tokens with more sampling steps",
            "mask": torch.zeros_like(codes),
            "sampling_steps": 20,
            "temperature": 0.8,
        },
        {
            "name": "coarse_only_regen",
            "description": "Regenerate only coarse tokens",
            "mask": torch.zeros_like(codes),
            "sampling_steps": 8,
            "temperature": 0.8,
        },
    ]
    
    # Set up masks
    tests[1]["mask"][:, :, 100:101] = 1  # Minimal mask
    tests[2]["mask"][:, 4:, :] = 1  # Fine tokens only
    tests[3]["mask"][:, 4:, :] = 1  # Fine tokens only
    tests[4]["mask"][:, 4:, :] = 1  # Fine tokens only
    tests[5]["mask"][:, :4, :] = 1  # Coarse tokens only
    
    for i, test in enumerate(tests):
        print(f"\n=== Test {i+1}: {test['name']} ===")
        print(f"Description: {test['description']}")
        print(f"Masked tokens: {test['mask'].sum().item()}")
        
        if test["sampling_steps"] == 0:
            # Direct decode
            output = codes
        else:
            # Use VampNet
            output = interface.vamp(
                codes.clone(),
                test["mask"],
                sampling_steps=test["sampling_steps"],
                temperature=test["temperature"],
                return_mask=False,
                typical_filtering=True,
                typical_mass=0.15,
            )
        
        # Check preservation
        unmasked_pos = (test["mask"] == 0)
        preserved = torch.all(codes[unmasked_pos] == output[unmasked_pos])
        print(f"Unmasked tokens preserved: {preserved}")
        
        # Decode and save
        output_signal = interface.decode(output)
        output_signal.write(f"output/diag_{test['name']}.wav")
        print(f"Saved: diag_{test['name']}.wav")
    
    # Also test without typical filtering
    print("\n=== Test without typical filtering ===")
    mask = torch.zeros_like(codes)
    mask[:, 4:, :] = 1
    
    output_no_typical = interface.vamp(
        codes.clone(),
        mask,
        sampling_steps=8,
        temperature=0.8,
        return_mask=False,
        typical_filtering=False,  # Disable typical filtering
    )
    
    signal_no_typical = interface.decode(output_no_typical)
    signal_no_typical.write("output/diag_no_typical_filtering.wav")
    print("Saved: diag_no_typical_filtering.wav")
    
    print("\n=== Diagnosis complete ===")
    print("\nListen to the files and compare:")
    print("- diag_direct_decode.wav should be identical to original")
    print("- diag_low_temp.wav should be more conservative")
    print("- diag_high_temp.wav should be more varied")
    print("- diag_coarse_only_regen.wav shows coarse model quality")
    print("- diag_no_typical_filtering.wav tests filtering impact")


if __name__ == "__main__":
    import os
    os.makedirs("output", exist_ok=True)
    diagnose_quality()