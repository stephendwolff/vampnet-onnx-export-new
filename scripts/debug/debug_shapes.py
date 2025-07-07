import torch
import vampnet_onnx as vampnet

# Create interface
interface = vampnet.interface.Interface.default()

# Create dummy tokens to test shape transformations
dummy_z = torch.randint(0, 1024, (1, 14, 100))  # (batch, codebooks, time)

print(f"Input z shape: {dummy_z.shape}")

# Test the shape transformations
embedding_out = interface.coarse.embedding.from_codes(dummy_z, interface.onnx_codec)
print(f"After embedding.from_codes: {embedding_out.shape}")

quantized_features = interface.onnx_codec.quantizer.from_latents(dummy_z)
print(f"After quantizer.from_latents: {quantized_features[0].shape}")