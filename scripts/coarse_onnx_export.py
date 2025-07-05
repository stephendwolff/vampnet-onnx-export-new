#!/usr/bin/env python3
"""
Export VampNet coarse model to ONNX format.

The coarse model uses only the first n_codebooks (typically 4) of the 14 total codebooks
produced by the LAC codec. This script exports the model to ONNX for inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from vampnet_onnx.modules.transformer import VampNet
from vampnet_onnx.lac_onnx import LAC_ONNX


def test_pytorch_model(model, codebook_tables):
    """Test the PyTorch model to understand its behavior"""
    model.eval()
    
    # Create test input
    batch_size = 1
    time_steps = 10  # Small for testing
    codes = torch.randint(0, model.vocab_size, (batch_size, model.n_codebooks, time_steps))
    
    # Try to run through the model
    with torch.no_grad():
        # Create a mock codec with the codebook tables
        class MockCodec:
            class Quantizer:
                def __init__(self, tables):
                    self.quantizers = []
                    for table in tables:
                        mock_q = type('MockQuantizer', (), {
                            'codebook': type('Codebook', (), {'weight': table})()
                        })()
                        self.quantizers.append(mock_q)
            
            def __init__(self, tables):
                self.quantizer = self.Quantizer(tables)
        
        mock_codec = MockCodec(codebook_tables[:model.n_codebooks])
        
        # Get embeddings
        z_embed = model.embedding.from_codes(codes, mock_codec)
        print(f"Embeddings shape: {z_embed.shape}")
        
        # Project
        z_proj = model.embedding(z_embed)
        print(f"Projected shape: {z_proj.shape}")
        
        # Run through forward (which handles everything internally)
        logits = model.forward(z_embed)
        print(f"Logits shape: {logits.shape}")
        
    return logits


def export_coarse_vampnet_simple(model_path, output_path):
    """Export just the embedding lookup part first"""
    
    # Load the model
    print(f"Loading model from {model_path}")
    model = VampNet.load(location=Path(model_path), map_location="cpu", strict=False)
    model.eval()
    
    # Print model configuration
    print(f"Model configuration:")
    print(f"  n_codebooks: {model.n_codebooks}")
    print(f"  n_conditioning_codebooks: {model.n_conditioning_codebooks}")
    print(f"  vocab_size: {model.vocab_size}")
    print(f"  embedding_dim: {model.embedding_dim}")
    print(f"  n_heads: {model.n_heads}")
    print(f"  n_layers: {model.n_layers}")
    
    # Load codebook tables
    codebook_tables_path = Path(__file__).parent.parent / "lac_codebook_tables.pth"
    codebook_tables = torch.load(codebook_tables_path, map_location="cpu")
    
    # Test the PyTorch model first
    print("\nTesting PyTorch model...")
    test_pytorch_model(model, codebook_tables)
    
    # Create a simplified model for ONNX export
    class EmbeddingLookupONNX(nn.Module):
        def __init__(self, vampnet_model, codebook_tables, n_codebooks):
            super().__init__()
            self.n_codebooks = n_codebooks
            self.latent_dim = vampnet_model.latent_dim
            self.vocab_size = vampnet_model.vocab_size
            
            # Store codebook tables as buffers
            for i in range(n_codebooks):
                self.register_buffer(f'codebook_{i}', codebook_tables[i])
            
            # Check for special tokens
            if hasattr(vampnet_model.embedding, 'special'):
                self.has_special = True
                # Store special embeddings
                for token_name, param in vampnet_model.embedding.special.items():
                    self.register_buffer(f'special_{token_name}', param)
                self.special_names = list(vampnet_model.embedding.special.keys())
                self.mask_token_id = vampnet_model.mask_token
            else:
                self.has_special = False
                
        def forward(self, codes):
            """
            Convert codes to latents
            Args:
                codes: (batch, n_codebooks, time)
            Returns:
                latents: (batch, n_codebooks * latent_dim, time)
            """
            batch_size, n_codebooks, seq_len = codes.shape
            latents = []
            
            for i in range(self.n_codebooks):
                c = codes[:, i, :]  # (batch, time)
                
                # Get codebook for this layer
                codebook = getattr(self, f'codebook_{i}')  # (vocab_size, latent_dim)
                
                # Handle special tokens if needed
                if self.has_special:
                    # Build lookup table with special tokens
                    special_embeds = []
                    for name in self.special_names:
                        special_param = getattr(self, f'special_{name}')
                        special_embeds.append(special_param[i:i+1])  # (1, latent_dim)
                    
                    special_embeds = torch.cat(special_embeds, dim=0)  # (n_special, latent_dim)
                    lookup_table = torch.cat([codebook, special_embeds], dim=0)
                else:
                    lookup_table = codebook
                
                # Embedding lookup
                l = F.embedding(c, lookup_table)  # (batch, time, latent_dim)
                l = l.transpose(1, 2)  # (batch, latent_dim, time)
                latents.append(l)
            
            # Concatenate all latents
            latents = torch.cat(latents, dim=1)  # (batch, n_codebooks * latent_dim, time)
            return latents
    
    # Create embedding model
    embed_model = EmbeddingLookupONNX(model, codebook_tables, model.n_codebooks)
    embed_model.eval()
    
    # Test it
    print("\nTesting embedding lookup model...")
    dummy_codes = torch.randint(0, model.vocab_size, (1, model.n_codebooks, 10))
    with torch.no_grad():
        latents = embed_model(dummy_codes)
        print(f"Latents shape: {latents.shape}")
    
    # Export to ONNX
    embed_onnx_path = output_path.parent / "vampnet_coarse_embeddings.onnx"
    print(f"\nExporting embeddings to {embed_onnx_path}...")
    
    with torch.no_grad():
        torch.onnx.export(
            embed_model,
            dummy_codes,
            embed_onnx_path,
            input_names=["codes"],
            output_names=["latents"],
            dynamic_axes={
                "codes": {0: "batch", 2: "time"},
                "latents": {0: "batch", 2: "time"}
            },
            opset_version=17,
            do_constant_folding=True,
            export_params=True,
        )
    
    print(f"Exported embeddings to {embed_onnx_path}")
    
    # Now export the rest of the model
    class VampNetTransformerONNX(nn.Module):
        def __init__(self, vampnet_model):
            super().__init__()
            self.embedding = vampnet_model.embedding
            self.transformer = vampnet_model.transformer
            self.classifier = vampnet_model.classifier
            self.n_predict_codebooks = vampnet_model.n_predict_codebooks
            
        def forward(self, latents):
            """
            Process latents through transformer
            Args:
                latents: (batch, n_codebooks * latent_dim, time)
            Returns:
                logits: (batch, n_predict_codebooks * vocab_size, time)
            """
            # Project to embedding dimension
            x = self.embedding(latents)  # (batch, embedding_dim, time)
            
            # Create mask 
            x_mask = torch.ones(x.shape[0], x.shape[2], dtype=torch.bool, device=x.device)
            
            # Rearrange for transformer: (batch, time, embedding_dim)
            x = x.transpose(1, 2)
            
            # Pass through transformer
            out = self.transformer(x=x, x_mask=x_mask)
            
            # Rearrange back: (batch, embedding_dim, time)
            out = out.transpose(1, 2)
            
            # Project to vocabulary
            logits = self.classifier(out, None)
            
            # The classifier outputs all codebooks, need to reshape
            # from (batch, n_predict_codebooks * vocab_size, time)
            # This is already the expected format
            
            return logits
    
    # Create transformer model
    trans_model = VampNetTransformerONNX(model)
    trans_model.eval()
    
    # Test it
    print("\nTesting transformer model...")
    with torch.no_grad():
        # Use latents from embedding model
        latents = embed_model(dummy_codes)
        logits = trans_model(latents)
        print(f"Logits shape: {logits.shape}")
    
    # Export transformer
    trans_onnx_path = output_path.parent / "vampnet_coarse_transformer.onnx"
    print(f"\nExporting transformer to {trans_onnx_path}...")
    
    # Create dummy latents for export
    dummy_latents = torch.randn(1, model.n_codebooks * model.latent_dim, 10)
    
    with torch.no_grad():
        torch.onnx.export(
            trans_model,
            dummy_latents,
            trans_onnx_path,
            input_names=["latents"],
            output_names=["logits"],
            dynamic_axes={
                "latents": {0: "batch", 2: "time"},
                "logits": {0: "batch", 2: "time"}
            },
            opset_version=17,
            do_constant_folding=True,
            export_params=True,
        )
    
    print(f"Exported transformer to {trans_onnx_path}")
    
    # Verify the exports
    import onnxruntime as ort
    print(f"\nVerifying ONNX models...")
    
    # Test embedding model
    embed_session = ort.InferenceSession(str(embed_onnx_path))
    embed_out = embed_session.run(None, {"codes": dummy_codes.numpy()})[0]
    print(f"Embedding ONNX output shape: {embed_out.shape}")
    
    # Test transformer model  
    trans_session = ort.InferenceSession(str(trans_onnx_path))
    trans_out = trans_session.run(None, {"latents": embed_out})[0]
    print(f"Transformer ONNX output shape: {trans_out.shape}")
    
    print("\nExport complete! Two models created:")
    print(f"1. Embeddings: {embed_onnx_path}")
    print(f"2. Transformer: {trans_onnx_path}")


if __name__ == "__main__":
    # Default paths
    model_path = Path(__file__).parent.parent / "models" / "vampnet" / "coarse.pth"
    output_path = Path(__file__).parent.parent / "vampnet_coarse.onnx"
    
    # Check if model exists
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Please ensure the model file is in the correct location")
        sys.exit(1)
    
    export_coarse_vampnet_simple(model_path, output_path)