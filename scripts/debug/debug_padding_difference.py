import torch
import numpy as np
import vampnet_onnx as vampnet
import audiotools as at
from lac.model.lac import LAC

# Load models
pytorch_model = LAC.load("models/vampnet/codec.pth")
pytorch_model.eval()
interface = vampnet.interface.Interface.default()

# Load test audio
signal = at.AudioSignal("assets/example.wav")
original_length = signal.samples.shape[-1]
print(f"Original audio length: {original_length} samples")

# Check preprocessing in both
print("\n=== PyTorch Preprocessing ===")
pytorch_signal = signal.clone().resample(pytorch_model.sample_rate).to_mono()
pytorch_processed, pytorch_length = pytorch_model.preprocess(
    pytorch_signal.samples, pytorch_signal.sample_rate
)
print(f"After preprocess: {pytorch_processed.shape}, original length stored: {pytorch_length}")

print("\n=== ONNX Preprocessing ===")
onnx_signal = signal.clone().resample(interface.onnx_codec.sample_rate).to_mono()
onnx_processed, onnx_length = interface.onnx_codec.preprocess(
    onnx_signal.samples, onnx_signal.sample_rate
)
print(f"After preprocess: {onnx_processed.shape}, original length stored: {onnx_length}")

# Check hop length
print(f"\nHop length: {pytorch_model.hop_length}")
print(f"Expected frames: {original_length / pytorch_model.hop_length:.2f}")

# The issue might be in how we handle the signal in interface._preprocess
print("\n=== Interface Preprocessing ===")
interface_signal = signal.clone()
interface_preprocessed = interface._preprocess(interface_signal)
print(f"Interface preprocessed shape: {interface_preprocessed.samples.shape}")

# Let's trace through the exact encode path
print("\n=== Full Encode Comparison ===")

# PyTorch encode
with torch.no_grad():
    pt_encoded = pytorch_model.encode(pytorch_processed, pytorch_model.sample_rate)
    print(f"PyTorch encoded codes shape: {pt_encoded['codes'].shape}")

# ONNX encode (through interface)
onnx_codes = interface.encode(signal)
print(f"ONNX encoded codes shape: {onnx_codes.shape}")

# The difference is likely in the normalization or ensure_max_of_audio step
print("\n=== Checking Normalization ===")
test_signal = signal.clone()
print(f"Before normalization: min={test_signal.samples.min():.4f}, max={test_signal.samples.max():.4f}")
test_signal = test_signal.normalize(-24.0)
print(f"After normalize(-24): min={test_signal.samples.min():.4f}, max={test_signal.samples.max():.4f}")
test_signal = test_signal.ensure_max_of_audio(1.0)
print(f"After ensure_max(1.0): min={test_signal.samples.min():.4f}, max={test_signal.samples.max():.4f}")