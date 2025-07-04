from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
from lac.model.lac import LAC as DAC


class CodeExtractorFromZ(torch.nn.Module):
    """Extract codes from quantized z using pre-saved codebooks"""

    def __init__(self, codebooks_path):
        super().__init__()

        # Load pre-saved codebooks
        codebooks = torch.load(codebooks_path, map_location="cpu")

        # Store as parameters for ONNX export
        self.codebooks = nn.ParameterList(
            [nn.Parameter(cb.clone(), requires_grad=False) for cb in codebooks]
        )

        self.n_codebooks = len(codebooks)
        self.codebook_dim = codebooks[0].shape[1]  # Should be 8

        print(f"Loaded {self.n_codebooks} codebooks with dim {self.codebook_dim}")

    def forward(self, quantized_z):
        """
        Extract codes by finding nearest codebook entries
        quantized_z: (batch, features, time) - output from quantizer
        """
        batch_size, total_features, time_frames = quantized_z.shape
        print(f"Input shape: {quantized_z.shape}")
        print(f"Expected features per codebook: {self.codebook_dim}")
        print(f"Total features: {total_features}, n_codebooks: {self.n_codebooks}")

        # Check if the dimensions make sense
        expected_total = self.n_codebooks * self.codebook_dim
        if total_features != expected_total:
            raise ValueError(
                f"Expected {expected_total} features ({self.n_codebooks}*{self.codebook_dim}), got {total_features}"
            )

        # Reshape to separate codebooks: (batch, n_codebooks, codebook_dim, time)
        z_reshaped = quantized_z.view(
            batch_size, self.n_codebooks, self.codebook_dim, time_frames
        )

        codes = []
        for i in range(self.n_codebooks):
            z_book = z_reshaped[:, i, :, :]  # (batch, codebook_dim, time)
            codebook = self.codebooks[i]  # (vocab_size, codebook_dim)

            # Reshape for distance computation
            z_flat = z_book.permute(0, 2, 1).reshape(
                -1, self.codebook_dim
            )  # (batch*time, codebook_dim)

            # Compute distances and find nearest
            distances = torch.cdist(z_flat, codebook)  # (batch*time, vocab_size)
            codes_flat = torch.argmin(distances, dim=1)  # (batch*time,)

            # Reshape back
            codes_book = codes_flat.view(batch_size, time_frames)  # (batch, time)
            codes.append(codes_book)

        # Stack all codebooks
        codes_tensor = torch.stack(codes, dim=1)  # (batch, n_codebooks, time)
        return codes_tensor


# Test with actual dimensions first
def test_code_extractor_shapes():
    """Debug the actual shapes before ONNX export"""

    # Test with your actual quantizer output shape
    test_input = torch.randn(1, 1024, 114)  # Your actual quantizer output

    # code_extractor = CodeExtractorFromZ("codebook_tables.pth")
    code_extractor = CodeExtractorFromZ("lac_codebook_tables.pth")

    try:
        codes = code_extractor(test_input)
        print(f"✓ Success! Output codes shape: {codes.shape}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


# Run the test first
test_code_extractor_shapes()

# Export this as separate ONNX model
# lookup_tables = extract_codebook_lookup_tables(lac_model, "lac_codebook_tables.pth")
# codec_ckpt = (Path(__file__).parent / "../lac_codebook_tables.pth").resolve()

code_extractor = CodeExtractorFromZ("lac_codebook_tables.pth")
dummy_z = torch.randn(1, 1024, 114)  # Adjust to your quantizer output shape

torch.onnx.export(
    code_extractor,
    dummy_z,
    "code_extractor.onnx",
    export_params=True,
    opset_version=17,
    input_names=["quantized_z"],
    output_names=["codes"],
    dynamic_axes={
        "quantized_z": {0: "batch_size", 2: "time_frames"},
        "codes": {0: "batch_size", 2: "time_frames"},
    },
)
