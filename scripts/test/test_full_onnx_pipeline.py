#!/usr/bin/env python3
"""
Test the full ONNX pipeline: encode -> VampNet -> decode
"""

import torch
import audiotools as at
import vampnet_onnx as vampnet
import logging

# Enable info logging
logging.basicConfig(level=logging.INFO)


def test_full_pipeline():
    """Test the complete ONNX pipeline"""
    
    print("Loading ONNX interface...")
    interface = vampnet.interface.Interface.default(use_onnx=True)
    
    # Load test audio
    print("\nLoading test audio...")
    signal = at.AudioSignal("assets/example.wav")
    signal.write("output/original.wav")
    print(f"Original audio: {signal.duration} seconds, {signal.samples.shape}")
    
    # Encode to codes
    print("\nEncoding audio to codes...")
    codes = interface.encode(signal)
    print(f"Encoded codes shape: {codes.shape}")
    
    # Create a simple mask (mask middle portion)
    mask = torch.zeros_like(codes)
    mask[:, :, codes.shape[2]//4:3*codes.shape[2]//4] = 1
    print(f"Mask shape: {mask.shape}")
    print(f"Masked time steps: {mask[0, 0, :].sum().item()}/{codes.shape[2]}")
    
    # Test coarse generation
    print("\n1. Testing coarse VampNet...")
    coarse_codes = interface.coarse_vamp(
        z=codes,
        mask=mask,
        sampling_steps=8,
        temperature=0.8
    )
    print(f"Coarse output shape: {coarse_codes.shape}")
    
    # Test coarse-to-fine
    if interface.c2f is not None:
        print("\n2. Testing coarse-to-fine...")
        fine_codes = interface.coarse_to_fine(
            coarse_codes,
            sampling_steps=4,
            temperature=0.8
        )
        print(f"Fine output shape: {fine_codes.shape}")
    else:
        print("\n2. C2F model not available, using coarse codes only")
        fine_codes = coarse_codes
    
    # Decode back to audio
    print("\n3. Decoding to audio...")
    output_signal = interface.decode(fine_codes)
    print(f"Output audio: {output_signal.duration} seconds, {output_signal.samples.shape}")
    
    # Save output
    output_signal.write("output/vampnet_onnx_output.wav")
    print("\nOutput saved to: output/vampnet_onnx_output.wav")
    
    # Compare with original encoding/decoding
    print("\n4. Testing direct encode/decode (no VampNet)...")
    direct_decoded = interface.decode(codes)
    direct_decoded.write("output/direct_decoded.wav")
    print("Direct decoded saved to: output/direct_decoded.wav")
    
    print("\nPipeline test complete!")
    print("Files created:")
    print("  - output/original.wav (input)")
    print("  - output/direct_decoded.wav (encode->decode)")
    print("  - output/vampnet_onnx_output.wav (encode->vampnet->decode)")


if __name__ == "__main__":
    # Create output directory
    import os
    os.makedirs("output", exist_ok=True)
    
    # Run test
    test_full_pipeline()