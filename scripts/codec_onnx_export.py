import torch
import torch.onnx
import torch.nn as nn
from pathlib import Path

# from dac.model.dac import DAC
from lac.model.lac import LAC as DAC


# Register Snake1d symbolic function
class Snake1d_ONNX_Compatible(nn.Module):
    """ONNX-compatible Snake1d with channel-wise alpha parameters"""

    def __init__(self, alpha_tensor):
        super().__init__()
        # Store the full alpha tensor as a parameter
        self.register_parameter("alpha", nn.Parameter(alpha_tensor.clone()))

    def forward(self, x):
        # x shape: (batch, channels, time)
        # alpha shape: (1, channels, 1)
        # Snake: x + (1/alpha) * sin(alpha * x)^2
        alpha_expanded = self.alpha  # Broadcasting will handle the shapes
        return x + (1.0 / alpha_expanded) * torch.pow(torch.sin(alpha_expanded * x), 2)


def replace_snake1d_modules(model):
    """Replace all Snake1d modules with ONNX-compatible versions"""
    for name, module in model.named_children():
        if module.__class__.__name__ == "Snake1d":
            # Extract the full alpha tensor
            alpha_tensor = module.alpha.data.clone()
            print(f"Replacing {name} Snake1d with alpha shape={alpha_tensor.shape}")
            setattr(model, name, Snake1d_ONNX_Compatible(alpha_tensor))
        else:
            replace_snake1d_modules(module)


class LAC_Encoder_ONNX(torch.nn.Module):
    def __init__(self, lac_model):
        super().__init__()
        self.encoder = lac_model.encoder  # _orig_mod.encoder

    def forward(self, audio_waveform):
        return self.encoder(audio_waveform)


class LAC_Decoder_ONNX(torch.nn.Module):
    def __init__(self, lac_model):
        super().__init__()
        self.decoder = lac_model.decoder  # _orig_mod.decoder

    def forward(self, encoded_features):
        return self.decoder(encoded_features)


# Load your LAC model
codec_ckpt = (Path(__file__).parent / "../models/vampnet/codec.pth").resolve()
device = "cpu"
lac_model = DAC.load(codec_ckpt)
lac_model.to(device)
lac_model.eval()

print("Replacing Snake1d modules...")
replace_snake1d_modules(lac_model)
print("Snake1d replacement completed.")

# Audio parameters
sample_rate = 44100
duration = 2.0
samples = int(sample_rate * duration)
time_frames = samples // 6144  # Your total downsampling factor

print(f"Input samples: {samples}")
print(f"Expected output time frames: {time_frames}")

# Export Encoder
print("Exporting encoder...")
encoder_wrapper = LAC_Encoder_ONNX(lac_model)
encoder_wrapper.eval()

dummy_audio = torch.randn(1, 1, samples)

torch.onnx.export(
    encoder_wrapper,
    dummy_audio,
    "models_onnx/lac_encoder.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["audio_waveform"],
    output_names=["encoded_features"],
    dynamic_axes={
        "audio_waveform": {0: "batch_size", 2: "audio_length"},
        "encoded_features": {0: "batch_size", 2: "time_frames"},
    },
    # verbose=False,
)

# Export Decoder
print("Exporting decoder...")
decoder_wrapper = LAC_Decoder_ONNX(lac_model)
decoder_wrapper.eval()

dummy_encoded = torch.randn(1, 1024, time_frames)

torch.onnx.export(
    decoder_wrapper,
    dummy_encoded,
    "models_onnx/lac_decoder.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["encoded_features"],
    output_names=["reconstructed_audio"],
    dynamic_axes={
        "encoded_features": {0: "batch_size", 2: "time_frames"},
        "reconstructed_audio": {0: "batch_size", 2: "audio_length"},
    },
    # verbose=False,
)

print("Export completed successfully!")
