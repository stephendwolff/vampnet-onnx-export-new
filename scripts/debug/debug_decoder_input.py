import torch
import onnxruntime as ort
import numpy as np

# Check what the quantizer actually outputs
quantizer_session = ort.InferenceSession("lac_quantizer.onnx")

print("Quantizer outputs:")
for i, out in enumerate(quantizer_session.get_outputs()):
    print(f"  Output {i}: {out.name} - shape: {out.shape}")

# Create a dummy continuous input 
dummy_continuous = np.random.randn(1, 1024, 100).astype(np.float32)

# Run the quantizer
outputs = quantizer_session.run(None, {"continuous_features": dummy_continuous})

print("\nActual output shapes:")
for i, out in enumerate(outputs):
    if isinstance(out, np.ndarray):
        print(f"  Output {i}: {out.shape}")
    else:
        print(f"  Output {i}: {type(out)}")

# The first output should be the full quantized features
quantized_z = outputs[0]
print(f"\nQuantized features shape: {quantized_z.shape}")
print(f"Expected by decoder: (batch, 1024, time)")

# Check if we can decode this
decoder_session = ort.InferenceSession("lac_decoder.onnx") 
try:
    decoded = decoder_session.run(None, {"encoded_features": quantized_z})
    print("✓ Decoder accepts quantized features from quantizer!")
except Exception as e:
    print(f"✗ Decoder error: {e}")