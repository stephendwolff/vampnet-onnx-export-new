#!/usr/bin/env python3
"""
Debug embedding lookup differences between PyTorch and ONNX
"""

import torch
import numpy as np
import vampnet_onnx as vampnet


def debug_embeddings():
    """Compare embedding lookups between PyTorch and ONNX"""
    
    print("Loading models...")
    # Load PyTorch model
    interface_pytorch = vampnet.interface.Interface.default(use_onnx=False)
    coarse_model = interface_pytorch.coarse
    
    # Load ONNX model
    interface_onnx = vampnet.interface.Interface.default(use_onnx=True)
    coarse_onnx = interface_onnx.coarse
    
    # Create test input - simple codes
    batch_size = 1
    n_codebooks = 4
    time_steps = 10
    
    # Create codes with known values
    codes = torch.zeros((batch_size, n_codebooks, time_steps), dtype=torch.long)
    # Set some specific values
    codes[0, 0, :5] = torch.tensor([10, 20, 30, 40, 50])
    codes[0, 1, :5] = torch.tensor([100, 200, 300, 400, 500])
    codes[0, 2, :5] = torch.tensor([0, 1, 2, 3, 4])
    codes[0, 3, :5] = torch.tensor([1024, 1024, 100, 200, 1024])  # Include mask tokens
    
    print(f"\nTest codes shape: {codes.shape}")
    print(f"Sample values: {codes[0, :, :5]}")
    
    # Get embeddings from PyTorch
    print("\n=== PyTorch Embeddings ===")
    with torch.no_grad():
        # Use the model's forward method
        embeddings_pytorch = coarse_model(codes)
        
        print(f"Embeddings shape: {embeddings_pytorch.shape}")
        print(f"Embeddings stats: mean={embeddings_pytorch.mean():.4f}, std={embeddings_pytorch.std():.4f}")
    
    # Get embeddings from ONNX
    print("\n=== ONNX Embeddings ===")
    codes_np = codes.cpu().numpy()
    latents_onnx = coarse_onnx.embed_session.run(None, {"codes": codes_np})[0]
    embeddings_onnx = torch.from_numpy(latents_onnx)
    
    print(f"Embeddings shape: {embeddings_onnx.shape}")
    print(f"Embeddings stats: mean={embeddings_onnx.mean():.4f}, std={embeddings_onnx.std():.4f}")
    
    # Compare
    print("\n=== Comparison ===")
    close = torch.allclose(embeddings_pytorch, embeddings_onnx, rtol=1e-3, atol=1e-5)
    print(f"Embeddings match: {close}")
    
    if not close:
        diff = torch.abs(embeddings_pytorch - embeddings_onnx)
        print(f"Max difference: {diff.max():.6f}")
        print(f"Mean difference: {diff.mean():.6f}")
        
        # Find where they differ most
        max_idx = torch.argmax(diff)
        max_idx_unraveled = np.unravel_index(max_idx.item(), diff.shape)
        print(f"Max diff at position: {max_idx_unraveled}")
        print(f"PyTorch value: {embeddings_pytorch[max_idx_unraveled]:.6f}")
        print(f"ONNX value: {embeddings_onnx[max_idx_unraveled]:.6f}")
    
    # Test transformer forward pass
    print("\n=== Testing Transformer ===")
    
    # PyTorch forward
    with torch.no_grad():
        logits_pytorch = coarse_model.transformer(embeddings_pytorch)
        print(f"PyTorch logits shape: {logits_pytorch.shape}")
        print(f"PyTorch logits stats: mean={logits_pytorch.mean():.4f}, std={logits_pytorch.std():.4f}")
    
    # ONNX forward
    logits_onnx_np = coarse_onnx.trans_session.run(None, {"latents": latents_onnx})[0]
    logits_onnx = torch.from_numpy(logits_onnx_np)
    print(f"ONNX logits shape: {logits_onnx.shape}")
    print(f"ONNX logits stats: mean={logits_onnx.mean():.4f}, std={logits_onnx.std():.4f}")
    
    # Compare logits
    logits_close = torch.allclose(logits_pytorch, logits_onnx, rtol=1e-2, atol=1e-4)
    print(f"\nLogits match: {logits_close}")
    
    if not logits_close:
        logits_diff = torch.abs(logits_pytorch - logits_onnx)
        print(f"Max logits difference: {logits_diff.max():.6f}")
        print(f"Mean logits difference: {logits_diff.mean():.6f}")


if __name__ == "__main__":
    debug_embeddings()