import torch
import torch.nn as nn
import torch.onnx
import onnxruntime as ort
from pathlib import Path
from lac.model.lac import LAC

class FromCodesModule(nn.Module):
    """Module that implements the from_codes functionality with out_proj layers"""
    
    def __init__(self, lac_model):
        super().__init__()
        self.quantizers = lac_model.quantizer.quantizers
        self.n_codebooks = len(self.quantizers)
        
        # Store the codebooks and out_proj layers
        self.codebooks = nn.ModuleList()
        self.out_projs = nn.ModuleList()
        
        for i, quantizer in enumerate(self.quantizers):
            # Copy the codebook embedding
            codebook = nn.Embedding(
                quantizer.codebook.num_embeddings,
                quantizer.codebook.embedding_dim
            )
            codebook.weight.data = quantizer.codebook.weight.data.clone()
            self.codebooks.append(codebook)
            
            # Copy the out_proj layer with proper weight normalization handling
            out_proj = nn.Conv1d(
                quantizer.out_proj.in_channels,
                quantizer.out_proj.out_channels,
                kernel_size=1,
                bias=True  # The original has bias
            )
            
            # Handle weight normalization
            if hasattr(quantizer.out_proj, 'weight_v') and hasattr(quantizer.out_proj, 'weight_g'):
                # Compute the normalized weight from weight_v and weight_g
                weight_v = quantizer.out_proj.weight_v
                weight_g = quantizer.out_proj.weight_g
                
                # Compute norm of weight_v
                norm = weight_v.norm(2, 1, keepdim=True)
                weight = weight_v * (weight_g / norm)
                
                out_proj.weight.data = weight.data.clone()
            else:
                # No weight norm, just copy the weight
                out_proj.weight.data = quantizer.out_proj.weight.data.clone()
            
            # Copy bias
            if hasattr(quantizer.out_proj, 'bias') and quantizer.out_proj.bias is not None:
                out_proj.bias.data = quantizer.out_proj.bias.data.clone()
                
            self.out_projs.append(out_proj)
            
        print(f"Initialized {self.n_codebooks} quantizers")
        print(f"Input dim per quantizer: {self.quantizers[0].codebook_dim}")
        print(f"Output dim per quantizer: {self.quantizers[0].out_proj.out_channels}")
        
    def forward(self, codes):
        """
        Convert codes to full quantized features using codebook lookup and projection
        codes: (batch, n_codebooks, time) - discrete codes from VampNet
        returns: (batch, 1024, time) - quantized features for decoder
        """
        batch_size, n_codebooks, time_frames = codes.shape
        
        # Initialize the output
        z_q = 0.0
        
        # Process each codebook
        for i in range(n_codebooks):
            # Get codes for this codebook
            codes_i = codes[:, i, :]  # (batch, time)
            
            # Decode using codebook lookup
            codes_i = codes_i.long()
            z_p_i = self.codebooks[i](codes_i)  # (batch, time, codebook_dim)
            z_p_i = z_p_i.transpose(1, 2)  # (batch, codebook_dim, time)
            
            # Project to full dimension using out_proj
            z_q_i = self.out_projs[i](z_p_i)  # (batch, full_dim, time)
            
            # Sum the contributions
            z_q = z_q + z_q_i
        
        return z_q


# Load the LAC model
print("Loading LAC model...")
lac_model = LAC.load("models/vampnet/codec.pth")
lac_model.eval()

# Create the from_codes module
from_codes_module = FromCodesModule(lac_model)

# Test with dummy codes
test_codes = torch.randint(0, 1024, (1, 14, 100), dtype=torch.long)
print(f"\nTest input codes shape: {test_codes.shape}")

# Test forward pass
with torch.no_grad():
    output = from_codes_module(test_codes)
    print(f"Output shape: {output.shape}")
    
    # Compare with original from_codes
    original_output = lac_model.quantizer.from_codes(test_codes)[0]
    print(f"Original output shape: {original_output.shape}")
    
    # Check if outputs match
    diff = torch.abs(output - original_output).max()
    print(f"Max difference: {diff.item():.6f}")
    if diff < 1e-5:
        print("✓ Outputs match!")
    else:
        print("✗ Outputs don't match!")

# Export to ONNX
print("\nExporting to ONNX...")
torch.onnx.export(
    from_codes_module,
    test_codes,
    "../lac_from_codes.onnx",
    export_params=True,
    opset_version=11,
    input_names=["codes"],
    output_names=["quantized_features"],
    dynamic_axes={
        "codes": {0: "batch_size", 2: "time_frames"},
        "quantized_features": {0: "batch_size", 2: "time_frames"}
    },
)

print("✓ Exported lac_from_codes.onnx successfully!")

# Test the ONNX model
session = ort.InferenceSession("../lac_from_codes.onnx")

print("\nONNX Model Info:")
print("Inputs:")
for inp in session.get_inputs():
    print(f"  {inp.name}: {inp.shape}")
print("Outputs:")
for out in session.get_outputs():
    print(f"  {out.name}: {out.shape}")

# Test inference
test_codes_np = test_codes.numpy()
onnx_output = session.run(None, {"codes": test_codes_np})[0]
print(f"\nONNX output shape: {onnx_output.shape}")

# Check accuracy
onnx_diff = torch.abs(torch.from_numpy(onnx_output) - original_output).max()
print(f"ONNX vs PyTorch max difference: {onnx_diff.item():.6f}")