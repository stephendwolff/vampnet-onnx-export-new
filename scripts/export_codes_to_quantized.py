import torch
import onnx
from lac.model.lac import LAC

# Load the original PyTorch model
model = LAC.load("models/vampnet/codec.pth")
model.eval()

# Create a wrapper module that goes from codes to quantized features
class CodesToQuantized(torch.nn.Module):
    def __init__(self, quantizer):
        super().__init__()
        self.quantizer = quantizer
        
    def forward(self, codes):
        """
        Convert discrete codes to full quantized features
        Args:
            codes: (batch, n_codebooks, time) discrete codes
        Returns:
            quantized_z: (batch, 1024, time) quantized features
        """
        batch_size, n_codebooks, time_frames = codes.shape
        
        # Initialize output tensor
        quantized_z = []
        
        # Process each codebook
        for i in range(n_codebooks):
            # Get codes for this codebook
            codes_i = codes[:, i, :]  # (batch, time)
            
            # Get codebook embeddings
            codebook = self.quantizer.quantizers[i].codebook
            quantized_i = codebook.decode(codes_i)  # (batch, dim, time)
            
            quantized_z.append(quantized_i)
        
        # Concatenate all quantized features
        quantized_z = torch.cat(quantized_z, dim=1)  # (batch, 1024, time)
        
        return quantized_z

# Create the wrapper
codes_to_quantized = CodesToQuantized(model.quantizer)

# Create dummy input
dummy_codes = torch.randint(0, 1024, (1, 14, 100), dtype=torch.long)

# Export to ONNX
torch.onnx.export(
    codes_to_quantized,
    dummy_codes,
    "lac_codes_to_quantized.onnx",
    input_names=["codes"],
    output_names=["quantized_features"],
    dynamic_axes={
        "codes": {0: "batch_size", 2: "time_frames"},
        "quantized_features": {0: "batch_size", 2: "time_frames"}
    },
    opset_version=11,
)

print("Exported codes_to_quantized.onnx successfully!")

# Test the export
import onnxruntime as ort

session = ort.InferenceSession("lac_codes_to_quantized.onnx")
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
print(f"\nTest output shape: {onnx_output.shape}")