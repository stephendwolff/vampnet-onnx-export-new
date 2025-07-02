# quantizer_export.py
import torch
import torch.onnx

from pathlib import Path

# from dac.model.dac import DAC
from lac.model.lac import LAC as DAC


def export_quantizer_only(output_path):
    """Export just the quantizer to ONNX"""

    # Load the model
    codec_ckpt = str((Path(__file__).parent / "../models/vampnet/codec.pth").resolve())
    device = "cpu"
    lac_model = DAC.load(codec_ckpt)
    lac_model.to(device)
    lac_model.eval()

    # Extract quantizer
    quantizer = lac_model.quantizer

    # Create dummy input (should match your encoder output)
    dummy_input = torch.randn(1, 1024, 114)  # Adjust based on actual encoder output

    # Test first
    with torch.no_grad():
        result = quantizer(dummy_input)
        print(f"Quantizer outputs: {list(result.keys())}")
        print(f"z shape: {result['z'].shape}")
        print(f"codes shape: {result['codes'].shape}")

    # Try export
    try:
        torch.onnx.export(
            quantizer,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["continuous_features"],
            output_names=["quantized_z"],  # May need to adjust based on what works
            dynamic_axes={
                "continuous_features": {
                    0: "batch_size",
                    2: "time_frames",
                },  # Dynamic time
                "quantized_z": {0: "batch_size", 2: "time_frames"},
            },
            verbose=True,
        )
        print(f"✓ Quantizer exported to {output_path}")
        return True
    except Exception as e:
        print(f"✗ Export failed: {e}")
        return False


# Usage
export_quantizer_only("lac_quantizer.onnx")
