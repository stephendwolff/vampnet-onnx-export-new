import torch
from lac.model.lac import LAC

# Load model
model = LAC.load("../models/vampnet/codec.pth")
model.eval()

# Check the out_proj layers
print("Checking out_proj layers:")
for i, q in enumerate(model.quantizer.quantizers):
    print(f"\nQuantizer {i}:")
    print(f"  out_proj type: {type(q.out_proj)}")
    print(f"  out_proj: {q.out_proj}")
    if hasattr(q.out_proj, 'weight'):
        print(f"  Weight shape: {q.out_proj.weight.shape}")
    break  # Just check the first one