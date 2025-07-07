import torch
from lac.model.lac import LAC

# Load the original PyTorch model
model = LAC.load("models/vampnet/codec.pth")
model.eval()

print("=== Investigating PyTorch LAC Model ===\n")

# Check the model structure
print("Model components:")
for name, module in model.named_children():
    print(f"  {name}: {type(module).__name__}")

print("\n=== Checking Quantizer Structure ===")
print(f"Quantizer type: {type(model.quantizer).__name__}")
print(f"Number of codebooks: {model.quantizer.n_codebooks}")

# Check if there's a method to go from codes to full decoder input
print("\n=== Quantizer Methods ===")
for method in dir(model.quantizer):
    if not method.startswith('_') and callable(getattr(model.quantizer, method)):
        print(f"  {method}")

# Let's trace through the decode process
print("\n=== Tracing Decode Process ===")

# Create dummy codes like VampNet would generate
dummy_codes = torch.randint(0, 1024, (1, 14, 100))
print(f"Input codes shape: {dummy_codes.shape}")

# Check if there's a method to convert codes to decoder input
if hasattr(model.quantizer, 'from_codes'):
    print("Found 'from_codes' method!")
    
if hasattr(model.quantizer, 'decode_code'):
    print("Found 'decode_code' method!")

# Let's see what the quantizer actually does
print("\n=== Testing Quantizer Forward Pass ===")
dummy_continuous = torch.randn(1, 1024, 100)
with torch.no_grad():
    quantized_out = model.quantizer(dummy_continuous)
    print(f"Quantizer returns {len(quantized_out)} outputs")
    for i, out in enumerate(quantized_out):
        if isinstance(out, torch.Tensor):
            print(f"  Output {i}: shape {out.shape}")
        else:
            print(f"  Output {i}: {type(out)}")

# The key question: How does the PyTorch model decode from codes?
print("\n=== Checking Decoder Input Requirements ===")
print(f"Decoder expects input channels: {model.decoder.model[0].in_channels}")

# Let's check if the model has a special method for decoding from codes
if hasattr(model, 'decode'):
    print("\nModel has 'decode' method")
    # Check its signature
    import inspect
    sig = inspect.signature(model.decode)
    print(f"decode signature: {sig}")