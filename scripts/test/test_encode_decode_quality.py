import torch
import numpy as np
import vampnet_onnx as vampnet
import audiotools as at
from lac.model.lac import LAC

# Load both PyTorch and ONNX models
print("Loading models...")
pytorch_model = LAC.load("models/vampnet/codec.pth")
pytorch_model.eval()

interface = vampnet.interface.Interface.default()

# Load test audio
signal = at.AudioSignal("assets/example.wav")
print(f"Original signal shape: {signal.samples.shape}")

# Test 1: PyTorch encode/decode
print("\n=== PyTorch Encode/Decode ===")
with torch.no_grad():
    # Preprocess
    pytorch_signal = signal.clone().resample(pytorch_model.sample_rate).to_mono()
    pytorch_signal.samples, length = pytorch_model.preprocess(pytorch_signal.samples, pytorch_signal.sample_rate)
    
    # Encode
    pytorch_encoded = pytorch_model.encode(pytorch_signal.samples, pytorch_signal.sample_rate)
    pytorch_codes = pytorch_encoded["codes"]
    print(f"PyTorch codes shape: {pytorch_codes.shape}")
    
    # Decode using from_codes
    quantized_full = pytorch_model.quantizer.from_codes(pytorch_codes)[0]
    print(f"PyTorch quantized shape: {quantized_full.shape}")
    
    # Decode to audio
    pytorch_decoded = pytorch_model.decode(quantized_full)
    pytorch_audio = at.AudioSignal(pytorch_decoded["audio"], pytorch_model.sample_rate)
    pytorch_audio.write("scratch/pytorch_decode.wav")

# Test 2: ONNX encode/decode
print("\n=== ONNX Encode/Decode ===")
onnx_codes = interface.encode(signal)
print(f"ONNX codes shape: {onnx_codes.shape}")

onnx_decoded = interface.decode(onnx_codes)
onnx_decoded.write("scratch/onnx_decode.wav")

# Compare codes
print("\n=== Comparing Codes ===")
print(f"Codes match: {torch.allclose(pytorch_codes, onnx_codes, atol=1e-3)}")
if not torch.allclose(pytorch_codes, onnx_codes):
    diff = torch.abs(pytorch_codes - onnx_codes).max()
    print(f"Max code difference: {diff}")

# Test 3: Check intermediate steps
print("\n=== Testing from_codes ONNX ===")
if interface.onnx_codec.from_codes_session is not None:
    # Get quantized features from ONNX
    onnx_quantized = interface.onnx_codec.from_codes_session.run(
        None, {"codes": onnx_codes.numpy()}
    )[0]
    print(f"ONNX from_codes output shape: {onnx_quantized.shape}")
    
    # Compare with PyTorch
    pytorch_quantized = pytorch_model.quantizer.from_codes(onnx_codes)[0]
    print(f"PyTorch from_codes output shape: {pytorch_quantized.shape}")
    
    diff = torch.abs(torch.from_numpy(onnx_quantized) - pytorch_quantized).max()
    print(f"Max quantized difference: {diff.item():.6f}")
    
    # Check a few values
    print("\nFirst few values comparison:")
    print(f"PyTorch: {pytorch_quantized[0, :5, 0].tolist()}")
    print(f"ONNX:    {onnx_quantized[0, :5, 0].tolist()}")
else:
    print("from_codes ONNX model not found!")

print("\n=== Summary ===")
print("Check the audio files:")
print("- scratch/pytorch_decode.wav (reference)")
print("- scratch/onnx_decode.wav (should sound similar)")