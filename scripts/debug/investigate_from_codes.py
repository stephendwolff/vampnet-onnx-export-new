import torch
from lac.model.lac import LAC

# Load the model
model = LAC.load("models/vampnet/codec.pth")
model.eval()

print("=== Investigating from_codes method ===\n")

# Create dummy codes
dummy_codes = torch.randint(0, 1024, (1, 14, 100))
print(f"Input codes shape: {dummy_codes.shape}")

# Try the from_codes method
with torch.no_grad():
    result = model.quantizer.from_codes(dummy_codes)
    print(f"\nfrom_codes returns: {type(result)}")
    if isinstance(result, torch.Tensor):
        print(f"Output shape: {result.shape}")
    elif isinstance(result, (list, tuple)):
        print(f"Number of outputs: {len(result)}")
        for i, out in enumerate(result):
            if isinstance(out, torch.Tensor):
                print(f"  Output {i}: shape {out.shape}")

# Now let's trace the full decode path
print("\n=== Full Decode Path ===")

# Test 1: Try to decode directly from codes
try:
    latents = model.quantizer.from_codes(dummy_codes)
    print(f"from_codes output shape: {latents.shape}")
    
    # Now try to decode these latents
    audio = model.decode(latents)
    print(f"✓ Direct decode from codes works! Audio shape: {audio.shape}")
except Exception as e:
    print(f"✗ Direct decode failed: {e}")

# Test 2: Check what from_latents does
print("\n=== Testing from_latents ===")
try:
    # from_codes gives us latents, let's see what from_latents expects
    latents = model.quantizer.from_codes(dummy_codes)
    quantized = model.quantizer.from_latents(latents)
    print(f"from_latents output type: {type(quantized)}")
    if isinstance(quantized, torch.Tensor):
        print(f"from_latents output shape: {quantized.shape}")
    elif isinstance(quantized, (list, tuple)):
        for i, out in enumerate(quantized):
            if isinstance(out, torch.Tensor):
                print(f"  Output {i}: shape {out.shape}")
except Exception as e:
    print(f"from_latents error: {e}")

# Let's check the actual implementation
print("\n=== Checking Implementation ===")
import inspect

# Get source of from_codes if possible
try:
    source = inspect.getsource(model.quantizer.from_codes)
    print("from_codes source preview:")
    lines = source.split('\n')[:10]
    for line in lines:
        print(f"  {line}")
except:
    print("Could not get source for from_codes")