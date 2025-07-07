#!/usr/bin/env python3
"""
Compare PyTorch and ONNX VampNet outputs to identify differences
"""

import torch
import audiotools as at
import vampnet_onnx as vampnet
import numpy as np


def compare_vampnet_implementations():
    """Compare PyTorch and ONNX implementations side by side"""
    
    # Load both interfaces
    print("Loading PyTorch interface...")
    interface_pytorch = vampnet.interface.Interface.default(use_onnx=False)
    
    print("Loading ONNX interface...")
    interface_onnx = vampnet.interface.Interface.default(use_onnx=True)
    
    # Load test audio
    print("\nLoading test audio...")
    signal = at.AudioSignal("assets/example.wav")
    
    # Encode with PyTorch (should be the same for both)
    print("Encoding audio...")
    codes = interface_pytorch.encode(signal)
    print(f"Codes shape: {codes.shape}")
    
    # Create test masks
    masks = {
        "middle_third": torch.zeros_like(codes),
        "fine_only": torch.zeros_like(codes),
        "periodic": torch.zeros_like(codes),
    }
    
    # Middle third mask
    start = codes.shape[2] // 3
    end = 2 * codes.shape[2] // 3
    masks["middle_third"][:, :, start:end] = 1
    
    # Fine only mask (keep coarse)
    masks["fine_only"][:, 4:, :] = 1
    
    # Periodic mask
    masks["periodic"][:, :, ::2] = 1
    
    # Test each mask
    for mask_name, mask in masks.items():
        print(f"\n=== Testing {mask_name} mask ===")
        masked_tokens = mask.sum().item()
        total_tokens = codes.numel()
        print(f"Masked tokens: {masked_tokens}/{total_tokens} ({masked_tokens/total_tokens*100:.1f}%)")
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Generate with PyTorch
        print("Generating with PyTorch...")
        output_pytorch = interface_pytorch.vamp(
            codes.clone(),
            mask.clone(),
            _sampling_steps=8,
            temperature=0.8,
            return_mask=False,
            seed=42
        )
        
        # Reset seed
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Generate with ONNX
        print("Generating with ONNX...")
        output_onnx = interface_onnx.vamp(
            codes.clone(),
            mask.clone(),
            sampling_steps=8,
            temperature=0.8,
            return_mask=False,
            seed=42
        )
        
        # Compare outputs
        print("\nComparing outputs:")
        
        # Check if unmasked tokens are preserved
        unmasked_positions = (mask == 0)
        pytorch_preserved = torch.all(codes[unmasked_positions] == output_pytorch[unmasked_positions])
        onnx_preserved = torch.all(codes[unmasked_positions] == output_onnx[unmasked_positions])
        print(f"PyTorch preserved unmasked: {pytorch_preserved}")
        print(f"ONNX preserved unmasked: {onnx_preserved}")
        
        # Check how many tokens differ
        pytorch_diff = (codes != output_pytorch).sum().item()
        onnx_diff = (codes != output_onnx).sum().item()
        print(f"PyTorch changed tokens: {pytorch_diff}")
        print(f"ONNX changed tokens: {onnx_diff}")
        
        # Check if outputs are identical
        outputs_match = torch.all(output_pytorch == output_onnx)
        print(f"Outputs identical: {outputs_match}")
        
        if not outputs_match:
            # Find where they differ
            diff_mask = (output_pytorch != output_onnx)
            diff_count = diff_mask.sum().item()
            print(f"Tokens that differ: {diff_count}")
            
            # Check if differences are only in masked regions
            diff_in_masked = torch.all(diff_mask[mask == 0] == False)
            print(f"Differences only in masked regions: {diff_in_masked}")
        
        # Decode and save
        signal_pytorch = interface_pytorch.decode(output_pytorch)
        signal_onnx = interface_onnx.decode(output_onnx)
        
        signal_pytorch.write(f"output/compare_{mask_name}_pytorch.wav")
        signal_onnx.write(f"output/compare_{mask_name}_onnx.wav")
        
        print(f"Saved: compare_{mask_name}_pytorch.wav and compare_{mask_name}_onnx.wav")
    
    # Also save the original
    signal.write("output/compare_original.wav")
    print("\nSaved original as compare_original.wav")
    
    print("\n=== Comparison complete ===")
    print("Listen to the output files to compare quality.")


if __name__ == "__main__":
    import os
    os.makedirs("output", exist_ok=True)
    compare_vampnet_implementations()