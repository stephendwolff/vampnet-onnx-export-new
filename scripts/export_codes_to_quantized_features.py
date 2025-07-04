import torch
import torch.nn as nn
import torch.onnx
import onnxruntime as ort
from pathlib import Path

class CodesToQuantizedFeatures(nn.Module):
    """Convert discrete codes to full quantized features using codebook lookup"""
    
    def __init__(self, codebooks_path):
        super().__init__()
        
        # Load pre-saved codebooks
        codebooks = torch.load(codebooks_path, map_location="cpu")
        
        # Store as parameters for ONNX export
        self.codebooks = nn.ParameterList(
            [nn.Parameter(cb.clone(), requires_grad=False) for cb in codebooks]
        )
        
        self.n_codebooks = len(codebooks)
        self.codebook_dim = codebooks[0].shape[1]  # Should be 8 for each
        self.vocab_size = codebooks[0].shape[0]    # Should be 1024
        
        print(f"Loaded {self.n_codebooks} codebooks")
        print(f"Codebook dim: {self.codebook_dim}, Vocab size: {self.vocab_size}")
        
    def forward(self, codes):
        """
        Convert codes to quantized features via codebook lookup
        codes: (batch, n_codebooks, time) - discrete codes from VampNet
        returns: (batch, 1024, time) - quantized features for decoder
        """
        batch_size, n_codebooks, time_frames = codes.shape
        
        # Collect quantized features from each codebook
        quantized_features = []
        
        for i in range(n_codebooks):
            # Get codes for this codebook
            codes_i = codes[:, i, :]  # (batch, time)
            
            # Get the codebook
            codebook = self.codebooks[i]  # (vocab_size, codebook_dim)
            
            # Lookup: gather the embeddings
            # codes_i needs to be long type for embedding
            codes_i = codes_i.long()
            
            # Reshape for embedding lookup
            codes_flat = codes_i.view(-1)  # (batch * time)
            
            # Lookup embeddings
            quantized_flat = F.embedding(codes_flat, codebook)  # (batch * time, codebook_dim)
            
            # Reshape back
            quantized_i = quantized_flat.view(batch_size, time_frames, self.codebook_dim)
            quantized_i = quantized_i.transpose(1, 2)  # (batch, codebook_dim, time)
            
            quantized_features.append(quantized_i)
        
        # Concatenate all codebook features
        quantized_z = torch.cat(quantized_features, dim=1)  # (batch, 1024, time)
        
        return quantized_z


# Test the module
def test_codes_to_quantized():
    converter = CodesToQuantizedFeatures("../lac_codebook_tables.pth")
    
    # Test input - discrete codes from VampNet
    test_codes = torch.randint(0, 1024, (1, 14, 100), dtype=torch.long)
    
    # Convert to quantized features
    quantized_features = converter(test_codes)
    print(f"Input codes shape: {test_codes.shape}")
    print(f"Output quantized features shape: {quantized_features.shape}")
    
    return converter, test_codes


# Export to ONNX
if __name__ == "__main__":
    import torch.nn.functional as F
    
    converter, dummy_codes = test_codes_to_quantized()
    
    # Export to ONNX
    torch.onnx.export(
        converter,
        dummy_codes,
        "../lac_codes_to_quantized.onnx",
        export_params=True,
        opset_version=11,
        input_names=["codes"],
        output_names=["quantized_features"],
        dynamic_axes={
            "codes": {0: "batch_size", 2: "time_frames"},
            "quantized_features": {0: "batch_size", 2: "time_frames"}
        },
    )
    
    print("\nExported lac_codes_to_quantized.onnx successfully!")
    
    # Test the ONNX model
    session = ort.InferenceSession("../lac_codes_to_quantized.onnx")
    
    print("\nONNX Model Info:")
    print("Inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.shape}")
    print("Outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.shape}")
    
    # Test inference
    test_codes = torch.randint(0, 1024, (1, 14, 50), dtype=torch.long)
    onnx_output = session.run(None, {"codes": test_codes.numpy()})[0]
    print(f"\nTest ONNX output shape: {onnx_output.shape}")