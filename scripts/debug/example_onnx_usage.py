#!/usr/bin/env python3
"""
Example script showing how to use VampNet with ONNX models
"""

import torch
import audiotools as at
from pathlib import Path
import logging

from vampnet_onnx.interface import Interface

logging.basicConfig(level=logging.INFO)


def main():
    # Create interface with ONNX models
    print("Loading VampNet with ONNX models...")
    
    try:
        # Try to load with ONNX models
        interface = Interface.default(use_onnx=True)
        print("✓ Loaded ONNX models successfully!")
    except FileNotFoundError as e:
        print(f"✗ ONNX models not found: {e}")
        print("\nTo generate ONNX models, run:")
        print("  python scripts/coarse_onnx_export.py")
        print("  python scripts/c2f_onnx_export.py")
        return
    
    # Load example audio
    audio_path = Path(__file__).parent / "assets" / "example.wav"
    if not audio_path.exists():
        print(f"Example audio not found at {audio_path}")
        return
        
    print(f"\nLoading audio from {audio_path}")
    sig = at.AudioSignal(str(audio_path))
    
    # Encode the audio
    print("Encoding audio...")
    z = interface.encode(sig)
    print(f"Encoded shape: {z.shape}")
    
    # Create a mask for vamping
    print("\nCreating mask...")
    mask = interface.build_mask(
        z=z,
        sig=sig,
        rand_mask_intensity=0.8,  # How much to mask
        prefix_s=0.5,  # Keep first 0.5 seconds
        suffix_s=0.5,  # Keep last 0.5 seconds
        periodic_prompt=7,  # Periodic unmasking
        periodic_prompt_width=1,
        upper_codebook_mask=3,  # Only mask first 3 codebooks
    )
    
    # Perform coarse vamping
    print("\nPerforming coarse generation...")
    zv, mask_z = interface.coarse_vamp(
        z,
        mask=mask,
        return_mask=True,
        temperature=0.8,
        _sampling_steps=12,  # Number of iterative refinement steps
    )
    print(f"Coarse output shape: {zv.shape}")
    
    # Perform coarse-to-fine if available
    if interface.c2f is not None:
        print("\nPerforming coarse-to-fine generation...")
        zv = interface.coarse_to_fine(
            zv,
            mask=mask,
            typical_filtering=True,
            _sampling_steps=3,
        )
        print(f"Fine output shape: {zv.shape}")
    
    # Decode back to audio
    print("\nDecoding to audio...")
    output_sig = interface.decode(zv)
    
    # Save the output
    output_path = Path(__file__).parent / "output_onnx_vamped.wav"
    output_sig.write(str(output_path))
    print(f"\n✓ Saved output to {output_path}")
    
    # Also save the masked version for comparison
    masked_sig = interface.decode(mask_z)
    masked_path = Path(__file__).parent / "output_onnx_masked.wav"
    masked_sig.write(str(masked_path))
    print(f"✓ Saved masked audio to {masked_path}")


if __name__ == "__main__":
    main()