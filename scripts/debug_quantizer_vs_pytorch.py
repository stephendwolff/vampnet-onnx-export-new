# Test if export worked

from pathlib import Path

import numpy as np
import torch
import torch.onnx
from lac.model.lac import LAC as DAC

import onnxruntime as ort


def debug_quantizer_vs_pytorch():
    """Compare PyTorch quantizer vs ONNX quantizer outputs"""

    # Load models
    codec_ckpt = str((Path(__file__).parent / "../models/vampnet/codec.pth").resolve())
    lac_model = DAC.load(codec_ckpt)
    lac_model.eval()

    # Test input
    test_input = torch.randn(1, 1024, 114)

    # PyTorch quantizer
    with torch.no_grad():
        pytorch_result = lac_model.quantizer(test_input)

    print("PyTorch quantizer outputs:")
    for key, value in pytorch_result.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

    # ONNX quantizer
    lac_quantizer_path = str(
        (Path(__file__).parent / "../lac_quantizer.onnx").resolve()
    )
    quantizer_session = ort.InferenceSession(lac_quantizer_path)
    onnx_result = quantizer_session.run(
        None, {"continuous_features": test_input.numpy()}
    )

    print(f"\nONNX quantizer output:")
    for i, output in enumerate(onnx_result):
        print(f"  output_{i}: {output.shape}")

    # Check if ONNX output matches PyTorch 'z'
    if len(onnx_result) == 1:
        pytorch_z = pytorch_result["z"].numpy()
        onnx_output = onnx_result[0]

        print(f"\nComparison:")
        print(f"  PyTorch 'z' shape: {pytorch_z.shape}")
        print(f"  ONNX output shape: {onnx_output.shape}")
        print(f"  Shapes match: {pytorch_z.shape == onnx_output.shape}")

        if pytorch_z.shape == onnx_output.shape:
            diff = np.abs(pytorch_z - onnx_output).max()
            print(f"  Max difference: {diff:.2e}")


debug_quantizer_vs_pytorch()
