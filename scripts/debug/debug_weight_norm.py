import torch
from lac.model.lac import LAC
import torch.nn.utils.weight_norm as weight_norm

# Load model
model = LAC.load("models/vampnet/codec.pth")
model.eval()

print("Checking weight normalization in quantizers...")

# Check the first quantizer
q = model.quantizer.quantizers[0]
print(f"\nQuantizer 0:")
print(f"out_proj type: {type(q.out_proj)}")
print(f"out_proj module: {q.out_proj}")

# Check if weight_norm is applied
if hasattr(q.out_proj, 'weight_v') and hasattr(q.out_proj, 'weight_g'):
    print("✓ Weight normalization is applied!")
    print(f"weight_v shape: {q.out_proj.weight_v.shape}")
    print(f"weight_g shape: {q.out_proj.weight_g.shape}")
    
    # The actual weight is computed as: weight = weight_v * (weight_g / ||weight_v||)
    with torch.no_grad():
        # Compute the normalized weight
        weight_v = q.out_proj.weight_v
        weight_g = q.out_proj.weight_g
        
        # Compute norm of weight_v
        norm = weight_v.norm(2, 1, keepdim=True)
        weight = weight_v * (weight_g / norm)
        
        print(f"Computed weight shape: {weight.shape}")
        print(f"Original weight_v norm: {norm[0].item():.4f}")
        print(f"weight_g value: {weight_g[0].item():.4f}")
else:
    print("✗ No weight normalization found")
    if hasattr(q.out_proj, 'weight'):
        print(f"Regular weight shape: {q.out_proj.weight.shape}")

# Let's also check the module list
print("\n\nModule structure:")
for name, param in q.out_proj.named_parameters():
    print(f"  {name}: {param.shape}")