import torch
from lac.model.lac import LAC

# Load the model
model = LAC.load("models/vampnet/codec.pth")
model.eval()

print("=== Testing PyTorch Decode Path ===\n")

# Create dummy codes like VampNet would generate
dummy_codes = torch.randint(0, 1024, (1, 14, 100))
print(f"Input codes shape: {dummy_codes.shape}")

# Use from_codes to get the decoder input
with torch.no_grad():
    outputs = model.quantizer.from_codes(dummy_codes)
    print(f"\nfrom_codes returns {len(outputs)} outputs:")
    for i, out in enumerate(outputs):
        print(f"  Output {i}: shape {out.shape}")
    
    # The first output should be the full quantized representation
    quantized_full = outputs[0]
    print(f"\nUsing first output for decoder: {quantized_full.shape}")
    
    # Try to decode
    try:
        audio = model.decode(quantized_full)
        print(f"✓ Decode successful! Audio shape: {audio.shape}")
    except Exception as e:
        print(f"✗ Decode failed: {e}")

# Now let's understand what from_codes actually does
print("\n=== Understanding from_codes ===")

# Check the quantizer structure
print(f"Number of quantizers: {len(model.quantizer.quantizers)}")
print(f"Each quantizer has dimension: {model.quantizer.quantizers[0].codebook.embedding_dim}")

# The key insight: from_codes must be reconstructing the full 1024-dim representation
# from the 14 codebooks. Let's verify this:
total_dim = sum(q.codebook.embedding_dim for q in model.quantizer.quantizers)
print(f"Total quantized dimensions: {total_dim}")
print(f"But decoder needs: 1024 dimensions")
print(f"Missing dimensions: {1024 - total_dim}")

print("\nConclusion: The PyTorch from_codes method must be doing something special")
print("to reconstruct the full 1024 dimensions from just the 14 codes!")