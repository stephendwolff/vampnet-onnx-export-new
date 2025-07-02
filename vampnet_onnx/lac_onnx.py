import math
import torch
import numpy as np
import onnxruntime as ort

from pathlib import Path
from audiotools import AudioSignal

# from dac.model.dac import DAC
from lac.model.lac import LAC as DAC


class LAC_ONNX:
    def __init__(
        self,
        encoder_path,
        quantizer_path,
        decoder_path,
        sample_rate=44100,
        hop_length=6144,
    ):
        self.encoder_session = ort.InferenceSession(encoder_path)
        self.quantizer = ort.InferenceSession(quantizer_path)
        self.decoder_session = ort.InferenceSession(decoder_path)

        self.sample_rate = sample_rate
        self.hop_length = hop_length

    def preprocess(self, audio_data, sample_rate):
        """Preprocess like original LAC"""
        if sample_rate != self.sample_rate:
            raise ValueError(f"Sample rate {sample_rate} != {self.sample_rate}")

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length

        if right_pad > 0:
            audio_data = torch.nn.functional.pad(audio_data, (0, right_pad))

        return audio_data, length

    def encode(self, audio, sample_rate):
        """Full ONNX encode: audio → codes"""
        # Handle AudioSignal
        if isinstance(audio, AudioSignal):
            audio_data = audio.audio_data
            sample_rate = audio.sample_rate
        else:
            audio_data = audio

        processed_audio, length = self.preprocess(audio_data, sample_rate)

        # 1. ONNX Encoder: audio → continuous features
        continuous = self.encoder_session.run(
            None, {"audio_waveform": processed_audio.numpy()}
        )[0]

        # 2. ONNX Quantizer: continuous → quantized + codes
        quantized = self.quantizer_session.run(
            None, {"continuous_features": continuous}
        )[0]

        return {
            "z": torch.from_numpy(quantized),
            "length": length,
            "continuous": torch.from_numpy(continuous),
        }

    def decode(self, z, length=None):
        """Decode from quantized features"""
        z_numpy = z.numpy() if isinstance(z, torch.Tensor) else z
        decoded = self.decoder_session.run(None, {"encoded_features": z_numpy})[0]
        audio_tensor = torch.from_numpy(decoded)

        if length is not None:
            audio_tensor = audio_tensor[..., :length]

        return {"audio": audio_tensor}
