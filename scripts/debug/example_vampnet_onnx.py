#!/usr/bin/env python3
"""
Example of using VampNet with ONNX models for audio generation.
"""

import torch
import audiotools as at
import vampnet_onnx as vampnet


def main():
    # Load the ONNX interface
    print("Loading VampNet ONNX interface...")
    interface = vampnet.interface.Interface.default(use_onnx=True)
    
    # Load audio
    print("\nLoading audio...")
    signal = at.AudioSignal("assets/example.wav")
    print(f"Input duration: {signal.duration:.2f} seconds")
    
    # Encode to discrete codes
    print("\nEncoding audio to discrete codes...")
    codes = interface.encode(signal)
    print(f"Codes shape: {codes.shape} (batch, codebooks, time)")
    
    # Create a mask for the middle portion
    mask = torch.zeros_like(codes)
    start = codes.shape[2] // 4
    end = 3 * codes.shape[2] // 4
    mask[:, :, start:end] = 1
    print(f"\nMasking time steps {start} to {end} ({end-start} steps)")
    
    # Generate with VampNet
    print("\nGenerating with VampNet (this may take a moment)...")
    output_codes = interface.vamp(
        codes,
        mask,
        sampling_steps=12,      # Number of iterative refinement steps
        temperature=0.8,        # Sampling temperature
        return_mask=False
    )
    
    # Decode back to audio
    print("\nDecoding to audio...")
    output_signal = interface.decode(output_codes)
    
    # Save the output
    output_signal.write("vampnet_output.wav")
    print(f"\nOutput saved to vampnet_output.wav")
    print(f"Output duration: {output_signal.duration:.2f} seconds")
    
    # You can also use individual stages:
    print("\n--- Individual Stages ---")
    
    # 1. Coarse generation only (faster, lower quality)
    print("\nCoarse generation only:")
    coarse_codes = interface.coarse_vamp(
        codes, 
        mask,
        sampling_steps=8,
        temperature=0.8
    )
    coarse_signal = interface.decode(coarse_codes)
    coarse_signal.write("vampnet_coarse_output.wav")
    print("Saved coarse output to vampnet_coarse_output.wav")
    
    # 2. Coarse-to-fine refinement (if c2f model is available)
    if interface.c2f is not None:
        print("\nCoarse-to-fine refinement:")
        fine_codes = interface.coarse_to_fine(
            coarse_codes,
            sampling_steps=4,
            temperature=0.8
        )
        fine_signal = interface.decode(fine_codes)
        fine_signal.write("vampnet_fine_output.wav")
        print("Saved refined output to vampnet_fine_output.wav")


if __name__ == "__main__":
    main()