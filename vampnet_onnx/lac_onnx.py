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
        codebook_tables_path,
        from_codes_path=None,
        sample_rate=44100,
        hop_length=6144,
    ):
        self.encoder_session = ort.InferenceSession(encoder_path)
        self.quantizer_session = ort.InferenceSession(quantizer_path)
        self.decoder_session = ort.InferenceSession(decoder_path)
        
        # Optional from_codes session for decoding from codes
        if from_codes_path and Path(from_codes_path).exists():
            self.from_codes_session = ort.InferenceSession(from_codes_path)
        else:
            self.from_codes_session = None

        self.codebook_tables = torch.load(codebook_tables_path, map_location="cpu")

        self.sample_rate = sample_rate
        self.hop_length = hop_length

        self.quantizer = self.ONNXQuantizer(self)

    class ONNXQuantizer:
        """ONNX quantizer that mimics PyTorch quantizer interface"""

        def __init__(self, parent_codec):
            self.parent = parent_codec
            self.quantizers = []
            for i, codebook_weight in enumerate(parent_codec.codebook_tables):
                mock_quantizer = self.MockQuantizer(codebook_weight)
                self.quantizers.append(mock_quantizer)
            # Cache for storing full quantized features
            self._cached_quantized_features = {}

        class MockQuantizer:
            """Mock individual quantizer with codebook.weight"""

            def __init__(self, codebook_weight):
                self.codebook = self.MockCodebook(codebook_weight)

            class MockCodebook:
                """Mock codebook with weight attribute"""

                def __init__(self, weight):
                    self.weight = weight

        def from_latents(self, z):
            """Convert discrete codes back to continuous latents"""
            if z.dtype in [torch.int64, torch.long]:
                batch_size, n_codebooks, time_frames = z.shape  # (1, 14, 576)

                # Manual codebook lookup to reconstruct continuous latents
                quantized_features = []
                for i in range(n_codebooks):
                    codebook_weight = self.quantizers[i].codebook.weight
                    codes_for_book = z[:, i, :].long()  # (batch, time)

                    # Lookup: (batch, time, feature_dim)
                    quantized_book = codebook_weight[codes_for_book]
                    quantized_features.append(quantized_book)

                # Concatenate all codebooks: (batch, time, total_feature_dim)
                continuous_latents = torch.cat(quantized_features, dim=-1)

                # Transpose to (batch, feature_dim, time) for decoder
                continuous_latents = continuous_latents.transpose(1, 2)

                return [continuous_latents]

            else:
                # z is already continuous latents
                return [z]

    def preprocess(self, audio_data, sample_rate):
        """Preprocess like original LAC"""
        if sample_rate != self.sample_rate:
            raise ValueError(f"Sample rate {sample_rate} != {self.sample_rate}")

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length

        if right_pad > 0:
            audio_data = torch.nn.functional.pad(audio_data, (0, right_pad))

        return audio_data, length

    def encode(self, audio, sample_rate=None, n_quantizers=None):
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
        quantizer_outputs = self.quantizer_session.run(
            None, {"continuous_features": continuous}
        )

        quantized_z = quantizer_outputs[0]  # (1, 1024, 114)
        all_codes = quantizer_outputs[1]  # (1, 14, 114) ← VampNet needs this!

        if n_quantizers is not None:
            # Trim codes to first n_quantizers
            codes = all_codes[:, :n_quantizers, :]
        else:
            # Use all codebooks (default)
            codes = all_codes

        # Cache the quantized features for this set of codes
        # Use a hash of the codes as the key
        codes_tensor = torch.from_numpy(codes)
        cache_key = hash(codes_tensor.cpu().numpy().tobytes())
        self.quantizer._cached_quantized_features[cache_key] = torch.from_numpy(quantized_z)

        out = {
            "length": length,
            "z": torch.from_numpy(quantized_z),
            "codes": codes_tensor,
            # Add other outputs for compatibility
            "latents": (
                torch.from_numpy(quantizer_outputs[2])
                if len(quantizer_outputs) > 2
                else None
            ),
        }

        return out

    def decode(self, z, length=None):
        """Decode from quantized features"""
        z_numpy = z.numpy() if isinstance(z, torch.Tensor) else z
        decoded = self.decoder_session.run(None, {"encoded_features": z_numpy})[0]
        audio_tensor = torch.from_numpy(decoded)

        if length is not None:
            audio_tensor = audio_tensor[..., :length]

        return {"audio": audio_tensor}
    
    def decode_from_codes(self, codes, length=None):
        """Decode from discrete codes by reconstructing full quantized features"""
        if self.from_codes_session is None:
            # First try to find cached quantized features
            cache_key = hash(codes.cpu().numpy().tobytes())
            if cache_key in self.quantizer._cached_quantized_features:
                quantized_z = self.quantizer._cached_quantized_features[cache_key]
            else:
                raise ValueError(
                    "Cannot decode from codes without lac_from_codes.onnx model. "
                    "Please provide from_codes_path when initializing LAC_ONNX."
                )
        else:
            # Use the from_codes ONNX model to reconstruct full features
            codes_numpy = codes.numpy() if isinstance(codes, torch.Tensor) else codes
            quantized_z = self.from_codes_session.run(None, {"codes": codes_numpy})[0]
            quantized_z = torch.from_numpy(quantized_z)
        
        return self.decode(quantized_z, length=length)
