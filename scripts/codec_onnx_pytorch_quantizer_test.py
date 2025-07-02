import torch
import numpy as np
import onnxruntime as ort

from pathlib import Path

# from dac.model.dac import DAC
from lac.model.lac import LAC as DAC


def test_onnx_encoder_with_pytorch_quantizer(lac_model, encoder_onnx_path):
    """Test ONNX encoder → PyTorch quantizer → PyTorch decoder"""

    # Load ONNX encoder
    encoder_session = ort.InferenceSession(encoder_onnx_path)

    # Test audio
    test_audio = torch.randn(1, 1, 88200)

    # 1. ONNX Encoder: audio → continuous features
    audio_numpy = test_audio.numpy()
    onnx_continuous = encoder_session.run(None, {"audio_waveform": audio_numpy})[0]

    # 2. PyTorch Quantizer: continuous → codes + quantized_z
    with torch.no_grad():
        continuous_tensor = torch.from_numpy(onnx_continuous)
        quant_result = lac_model.quantizer(continuous_tensor)

    # 3. PyTorch Decoder: quantized_z → audio
    with torch.no_grad():
        # Use original length (88200) for proper trimming
        decoded_result = lac_model.decode(quant_result["z"], length=88200)

    # Compare with full PyTorch pipeline
    with torch.no_grad():
        pytorch_result = lac_model(test_audio)

    # Get audio tensors
    onnx_hybrid_audio = decoded_result["audio"]
    pytorch_audio = pytorch_result["audio"]

    print(f"ONNX hybrid output shape: {onnx_hybrid_audio.shape}")
    print(f"PyTorch output shape: {pytorch_audio.shape}")

    # Trim both to shortest length for comparison
    min_length = min(onnx_hybrid_audio.shape[-1], pytorch_audio.shape[-1])
    onnx_trimmed = onnx_hybrid_audio[0, 0, :min_length]
    pytorch_trimmed = pytorch_audio[0, 0, :min_length]

    correlation = torch.corrcoef(torch.stack([onnx_trimmed, pytorch_trimmed]))[0, 1]

    print(f"ONNX encoder + PyTorch quantizer/decoder vs full PyTorch:")
    print(f"  Comparison length: {min_length} samples")
    print(f"  Correlation: {correlation:.6f}")
    print(f"  Codes shape: {quant_result['codes'].shape}")
    print(f"  Max difference: {torch.abs(onnx_trimmed - pytorch_trimmed).max():.2e}")

    return quant_result["codes"], correlation


codec_ckpt = str((Path(__file__).parent / "../models/vampnet/codec.pth").resolve())
device = "cpu"
lac_model = DAC.load(codec_ckpt)
lac_model.to(device)
lac_model.eval()

lac_encoder_onnx_path = (Path(__file__).parent / "../lac_encoder.onnx").resolve()
lac_decoder_onnx_path = (Path(__file__).parent / "../lac_decoder.onnx").resolve()
# Test it
# Test it
# codes, correlation = test_onnx_encoder_with_pytorch_quantizer(
#     lac_model, "lac_encoder.onnx"
# )

codes, correlation = test_onnx_encoder_with_pytorch_quantizer(
    lac_model, lac_encoder_onnx_path
)
print(codes)
