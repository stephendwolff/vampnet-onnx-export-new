#!/usr/bin/env python3
"""
Check mask token handling in ONNX models
"""

import torch
import numpy as np
import vampnet_onnx as vampnet


def check_mask_token():
    """Check how mask tokens are handled"""
    
    print("Loading interfaces...")
    interface_onnx = vampnet.interface.Interface.default(use_onnx=True)
    
    # Check mask token values
    print(f"\nCoarse model mask token: {interface_onnx.coarse.mask_token}")
    print(f"C2F model mask token: {interface_onnx.c2f.mask_token}")
    
    # Create simple test input
    batch = 1
    n_codebooks = 4
    time_steps = 10
    
    # Test codes with some mask tokens
    codes = torch.zeros((batch, n_codebooks, time_steps), dtype=torch.long)
    codes[0, 0, :5] = torch.tensor([10, 20, 1024, 40, 1024])  # Include mask tokens
    codes[0, 1, :5] = torch.tensor([100, 1024, 300, 1024, 500])
    
    print(f"\nTest codes with mask tokens (1024):")
    print(codes[0, :, :5])
    
    # Run through coarse model
    print("\n=== Testing Coarse Model ===")
    with torch.no_grad():
        # Get embeddings
        codes_np = codes.cpu().numpy()
        latents = interface_onnx.coarse.embed_session.run(None, {"codes": codes_np})[0]
        print(f"Embeddings shape: {latents.shape}")
        
        # Run transformer
        logits = interface_onnx.coarse.trans_session.run(None, {"latents": latents})[0]
        logits_torch = torch.from_numpy(logits)
        print(f"Logits shape: {logits_torch.shape}")
        print(f"Logits stats: mean={logits_torch.mean():.4f}, std={logits_torch.std():.4f}")
        
        # Check if mask tokens produce reasonable logits
        # Reshape to check specific positions
        logits_reshaped = logits_torch.permute(0, 2, 1).reshape(batch, time_steps, n_codebooks, 1024)
        
        # Check logits at mask token positions
        print("\nLogits at mask token positions:")
        print(f"Position (0,0,2) - mask token: max={logits_reshaped[0,2,0,:].max():.4f}, min={logits_reshaped[0,2,0,:].min():.4f}")
        print(f"Position (0,1,1) - mask token: max={logits_reshaped[0,1,1,:].max():.4f}, min={logits_reshaped[0,1,1,:].min():.4f}")
    
    # Test generation with a simple mask
    print("\n=== Testing Generation ===")
    mask = torch.zeros_like(codes)
    mask[0, 0, 2] = 1  # Mask one position
    mask[0, 1, 1] = 1  # Mask another position
    
    print(f"Mask positions: {torch.where(mask == 1)}")
    
    # Generate with minimal steps
    output = interface_onnx.coarse.generate(
        codec=interface_onnx.onnx_codec,
        time_steps=time_steps,
        start_tokens=codes,
        mask=mask,
        sampling_steps=2,
        temperature=0.8,
        return_signal=False
    )
    
    print(f"\nGenerated output shape: {output.shape}")
    print("Original vs Generated (first 5 time steps):")
    print("Original:")
    print(codes[0, :, :5])
    print("Generated:")
    print(output[0, :, :5])
    
    # Check if unmasked positions preserved
    unmasked = (mask == 0)
    preserved = torch.all(codes[unmasked] == output[unmasked])
    print(f"\nUnmasked tokens preserved: {preserved}")
    
    # Check if masked positions changed
    masked_positions = torch.where(mask == 1)
    for i in range(len(masked_positions[0])):
        b, c, t = masked_positions[0][i], masked_positions[1][i], masked_positions[2][i]
        orig = codes[b, c, t].item()
        gen = output[b, c, t].item()
        print(f"Masked position ({b},{c},{t}): {orig} -> {gen} (changed: {orig != gen})")


if __name__ == "__main__":
    check_mask_token()