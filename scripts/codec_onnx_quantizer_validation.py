# Test if export worked

from pathlib import Path

import numpy as np
import torch
import torch.onnx
from lac.model.lac import LAC as DAC


def validate_onnx_quantizer(onnx_path, pytorch_quantizer, test_input):
    """Validate ONNX quantizer matches PyTorch"""

    import onnxruntime as ort

    # Load ONNX quantizer
    quantizer_session = ort.InferenceSession(onnx_path)

    # PyTorch result
    with torch.no_grad():
        pytorch_result = pytorch_quantizer(test_input)

    # ONNX result
    onnx_result = quantizer_session.run(
        None, {"continuous_features": test_input.numpy()}
    )[0]

    # Compare
    pytorch_z = pytorch_result["z"].numpy()
    max_diff = np.abs(pytorch_z - onnx_result).max()

    print(f"Quantizer validation:")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  PyTorch shape: {pytorch_z.shape}")
    print(f"  ONNX shape: {onnx_result.shape}")

    return max_diff < 1e-3


codec_ckpt = str((Path(__file__).parent / "../models/vampnet/codec.pth").resolve())
device = "cpu"
lac_model = DAC.load(codec_ckpt)
lac_model.to(device)
lac_model.eval()

# Extract quantizer
quantizer = lac_model.quantizer

lac_quantizer_path = (Path(__file__).parent / "../lac_quantizer.onnx").resolve()

# Create dummy input (should match your encoder output)
dummy_input = torch.randn(1, 1024, 114)  # Adjust based on actual encoder output

validate_onnx_quantizer(lac_quantizer_path, quantizer, dummy_input)
