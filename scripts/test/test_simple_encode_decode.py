import vampnet_onnx as vampnet
import audiotools as at

# Load interface
interface = vampnet.interface.Interface.default()

# Load test audio
signal = at.AudioSignal("assets/example.wav")
print(f"Original signal: {signal.samples.shape}")

# Save original for comparison
signal.write("scratch/original.wav")

# Encode and decode
codes = interface.encode(signal)
print(f"Encoded codes shape: {codes.shape}")

# Decode back
decoded = interface.decode(codes)
print(f"Decoded signal: {decoded.samples.shape}")

# Save decoded
decoded.write("scratch/onnx_encoded_decoded.wav")

print("\nFiles saved:")
print("- scratch/original.wav")
print("- scratch/onnx_encoded_decoded.wav")
print("\nCompare these files to check quality")