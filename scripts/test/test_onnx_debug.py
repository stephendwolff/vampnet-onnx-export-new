#!/usr/bin/env python3
"""Debug ONNX shapes"""

import torch
import vampnet_onnx as vampnet
import logging

logging.basicConfig(level=logging.DEBUG)

# Load interface
interface = vampnet.interface.Interface.default(use_onnx=True)

# Create simple test case
batch = 1
n_codebooks = 4  # coarse uses 4
time_steps = 72  # typical chunk size

# Create test codes
codes = torch.randint(0, 1024, (batch, n_codebooks, time_steps))

# Create mask
mask = torch.zeros_like(codes)
mask[:, :, time_steps//4:3*time_steps//4] = 1

print(f"Input codes shape: {codes.shape}")
print(f"Mask shape: {mask.shape}")

# Test the coarse model directly
print("\nTesting coarse model generate...")
try:
    output = interface.coarse.generate(
        codec=interface.onnx_codec,
        time_steps=time_steps,
        start_tokens=codes,
        mask=mask,
        sampling_steps=2,  # Just 2 steps for debug
        temperature=0.8,
        return_signal=False
    )
    print(f"Success! Output shape: {output.shape}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()