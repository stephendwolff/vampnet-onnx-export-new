import torch
import numpy as np
import onnxruntime as ort

from pathlib import Path

# from dac.model.dac import DAC
from lac.model.lac import LAC as DAC


# quick_validate(lac_model, lac_encoder_onnx_path, lac_decoder_onnx_path)
def validate_lac_models(pytorch_model, encoder_path, decoder_path):
    """Simple validation that handles length differences automatically"""

    # Load ONNX models
    encoder_session = ort.InferenceSession(encoder_path)
    decoder_session = ort.InferenceSession(decoder_path)

    # Use a length that's a multiple of your downsampling factor (6144)
    # This minimizes padding issues
    test_length = 6144 * 14  # 86016 samples ≈ 1.95 seconds at 44.1kHz
    test_audio = torch.randn(1, 1, test_length)

    print(f"Testing with {test_length} samples...")

    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        if hasattr(pytorch_model, "encoder"):
            pytorch_encoded = pytorch_model.encoder(test_audio)
            pytorch_decoded = pytorch_model.decoder(pytorch_encoded)
        else:
            pytorch_encoded = pytorch_model._orig_mod.encoder(test_audio)
            pytorch_decoded = pytorch_model._orig_mod.decoder(pytorch_encoded)

    # ONNX inference
    audio_numpy = test_audio.numpy()
    onnx_encoded = encoder_session.run(None, {"audio_waveform": audio_numpy})[0]
    onnx_decoded = decoder_session.run(None, {"encoded_features": onnx_encoded})[0]

    # Compare models
    encoded_diff = np.abs(pytorch_encoded.numpy() - onnx_encoded).max()
    decoded_diff = np.abs(pytorch_decoded.numpy() - onnx_decoded).max()

    # Compare audio (trim to shortest length)
    pytorch_audio = pytorch_decoded[0, 0].numpy()
    onnx_audio = onnx_decoded[0, 0]
    min_length = min(len(pytorch_audio), len(onnx_audio))

    correlation = np.corrcoef(pytorch_audio[:min_length], onnx_audio[:min_length])[0, 1]

    # Results
    print(f"Encoder difference: {encoded_diff:.2e}")
    print(f"Decoder difference: {decoded_diff:.2e}")
    print(f"Audio correlation: {correlation:.6f}")
    print(f"Output lengths: PyTorch={len(pytorch_audio)}, ONNX={len(onnx_audio)}")

    # Simple pass/fail
    if encoded_diff < 1e-3 and decoded_diff < 1e-3 and correlation > 0.999:
        print("✅ EXPORT SUCCESSFUL")
    elif correlation > 0.99:
        print("⚠️ ACCEPTABLE (small numerical differences)")
    else:
        print("❌ EXPORT FAILED")

    return correlation > 0.99


codec_ckpt = str((Path(__file__).parent / "../models/vampnet/codec.pth").resolve())
device = "cpu"
lac_model = DAC.load(codec_ckpt)
lac_model.to(device)
lac_model.eval()

lac_encoder_onnx_path = (Path(__file__).parent / "../lac_encoder.onnx").resolve()
lac_decoder_onnx_path = (Path(__file__).parent / "../lac_decoder.onnx").resolve()


# Use it like this:
success = validate_lac_models(lac_model, lac_encoder_onnx_path, lac_decoder_onnx_path)
# "lac_encoder.onnx", "lac_decoder.onnx")
