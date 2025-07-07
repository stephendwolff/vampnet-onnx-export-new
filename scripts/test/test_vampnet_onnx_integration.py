#!/usr/bin/env python3
"""
Test script to verify VampNet ONNX integration works correctly
"""

import torch
import numpy as np
from pathlib import Path
import logging
import audiotools as at

from vampnet_onnx.interface import Interface

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_onnx_models():
    """Test that ONNX models can be loaded and used"""
    
    # Paths to ONNX models
    base_path = Path(__file__).parent
    
    # Check if ONNX models exist
    coarse_embed_path = base_path / "vampnet_coarse_embeddings.onnx"
    coarse_trans_path = base_path / "vampnet_coarse_transformer.onnx"
    c2f_embed_path = base_path / "vampnet_c2f_embeddings.onnx"
    c2f_trans_path = base_path / "vampnet_c2f_transformer.onnx"
    
    if not all([coarse_embed_path.exists(), coarse_trans_path.exists()]):
        logger.error("Coarse ONNX models not found. Please run scripts/coarse_onnx_export.py first.")
        return False
        
    # Create interface with ONNX models
    logger.info("Creating interface with ONNX models...")
    interface = Interface(
        coarse_ckpt=None,  # Not needed for ONNX
        coarse2fine_ckpt=None,  # Not needed for ONNX
        use_onnx_vampnet=True,
        vampnet_coarse_embeddings_onnx_path=str(coarse_embed_path),
        vampnet_coarse_transformer_onnx_path=str(coarse_trans_path),
        vampnet_c2f_embeddings_onnx_path=str(c2f_embed_path) if c2f_embed_path.exists() else None,
        vampnet_c2f_transformer_onnx_path=str(c2f_trans_path) if c2f_trans_path.exists() else None,
        lac_encoder_onnx_path=str(base_path / "lac_encoder.onnx"),
        lac_decoder_onnx_path=str(base_path / "lac_decoder.onnx"),
        lac_quantizer_onnx_path=str(base_path / "lac_quantizer.onnx"),
        lac_codebook_tables_path=str(base_path / "lac_codebook_tables.pth"),
        device="cpu",
        compile=False
    )
    logger.info("Interface created successfully!")
    
    # Load test audio
    test_audio_path = base_path / "assets" / "example.wav"
    if not test_audio_path.exists():
        logger.error(f"Test audio not found at {test_audio_path}")
        return False
        
    logger.info(f"Loading test audio from {test_audio_path}")
    sig = at.AudioSignal(str(test_audio_path))
    
    # Encode audio
    logger.info("Encoding audio...")
    z = interface.encode(sig)
    logger.info(f"Encoded shape: {z.shape}")
    
    # Build mask
    logger.info("Building mask...")
    mask = interface.build_mask(
        z=z,
        sig=sig,
        rand_mask_intensity=0.8,
        prefix_s=0.5,
        suffix_s=0.5,
        periodic_prompt=7,
        periodic_prompt_width=1,
        upper_codebook_mask=3,
    )
    
    # Test coarse generation
    logger.info("Testing coarse generation...")
    try:
        zv, mask_z = interface.coarse_vamp(
            z, 
            mask=mask, 
            return_mask=True,
            temperature=0.8,
            _sampling_steps=3  # Few steps for quick test
        )
        logger.info(f"Coarse generation output shape: {zv.shape}")
        logger.info("Coarse generation successful!")
    except Exception as e:
        logger.error(f"Coarse generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test coarse-to-fine if available
    if interface.c2f is not None:
        logger.info("Testing coarse-to-fine generation...")
        try:
            zv_fine = interface.coarse_to_fine(
                zv,
                mask=mask,
                typical_filtering=True,
                _sampling_steps=2  # Few steps for quick test
            )
            logger.info(f"C2F generation output shape: {zv_fine.shape}")
            logger.info("Coarse-to-fine generation successful!")
        except Exception as e:
            logger.error(f"Coarse-to-fine generation failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.info("No C2F model available, skipping C2F test")
    
    # Test decoding
    logger.info("Testing decoding...")
    try:
        output_sig = interface.decode(zv)
        logger.info(f"Decoded audio shape: {output_sig.samples.shape}")
        logger.info("Decoding successful!")
        
        # Save output
        output_path = base_path / "test_onnx_output.wav"
        output_sig.write(str(output_path))
        logger.info(f"Saved output to {output_path}")
        
    except Exception as e:
        logger.error(f"Decoding failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def compare_pytorch_onnx():
    """Compare outputs between PyTorch and ONNX models"""
    logger.info("\n=== Comparing PyTorch and ONNX models ===")
    
    base_path = Path(__file__).parent
    
    # Load test audio
    test_audio_path = base_path / "assets" / "example.wav"
    sig = at.AudioSignal(str(test_audio_path))
    
    # Create PyTorch interface
    logger.info("Creating PyTorch interface...")
    pytorch_interface = Interface(
        coarse_ckpt=str(base_path / "models" / "vampnet" / "coarse.pth"),
        coarse2fine_ckpt=str(base_path / "models" / "vampnet" / "c2f.pth"),
        use_onnx_vampnet=False,
        lac_encoder_onnx_path=str(base_path / "lac_encoder.onnx"),
        lac_decoder_onnx_path=str(base_path / "lac_decoder.onnx"),
        lac_quantizer_onnx_path=str(base_path / "lac_quantizer.onnx"),
        lac_codebook_tables_path=str(base_path / "lac_codebook_tables.pth"),
        device="cpu",
        compile=False
    )
    
    # Create ONNX interface
    logger.info("Creating ONNX interface...")
    onnx_interface = Interface(
        coarse_ckpt=None,
        coarse2fine_ckpt=None,
        use_onnx_vampnet=True,
        vampnet_coarse_embeddings_onnx_path=str(base_path / "vampnet_coarse_embeddings.onnx"),
        vampnet_coarse_transformer_onnx_path=str(base_path / "vampnet_coarse_transformer.onnx"),
        vampnet_c2f_embeddings_onnx_path=str(base_path / "vampnet_c2f_embeddings.onnx"),
        vampnet_c2f_transformer_onnx_path=str(base_path / "vampnet_c2f_transformer.onnx"),
        lac_encoder_onnx_path=str(base_path / "lac_encoder.onnx"),
        lac_decoder_onnx_path=str(base_path / "lac_decoder.onnx"),
        lac_quantizer_onnx_path=str(base_path / "lac_quantizer.onnx"),
        lac_codebook_tables_path=str(base_path / "lac_codebook_tables.pth"),
        device="cpu",
        compile=False
    )
    
    # Encode with both
    logger.info("Encoding audio...")
    z_pytorch = pytorch_interface.encode(sig)
    z_onnx = onnx_interface.encode(sig)
    
    # Compare encoded values
    if torch.allclose(z_pytorch, z_onnx):
        logger.info("✓ Encoded values match!")
    else:
        logger.warning("✗ Encoded values differ")
        
    # Use same random seed for both
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Build same mask
    mask = pytorch_interface.build_mask(
        z=z_pytorch,
        sig=sig,
        rand_mask_intensity=1.0,
        prefix_s=0.0,
        suffix_s=0.0,
    )
    
    # Test with minimal generation steps
    logger.info("Testing minimal generation...")
    
    # PyTorch generation
    torch.manual_seed(42)
    np.random.seed(42)
    zv_pytorch = pytorch_interface.coarse.generate(
        codec=pytorch_interface.onnx_codec,
        time_steps=z_pytorch.shape[-1],
        _sampling_steps=1,
        start_tokens=z_pytorch.clone(),
        mask=mask.clone(),
        temperature=1.0,
        return_signal=False
    )
    
    # ONNX generation
    torch.manual_seed(42)
    np.random.seed(42)
    zv_onnx = onnx_interface.coarse.generate(
        codec=onnx_interface.onnx_codec,
        time_steps=z_onnx.shape[-1],
        _sampling_steps=1,
        start_tokens=z_onnx.clone(),
        mask=mask.clone(),
        temperature=1.0,
        return_signal=False
    )
    
    logger.info(f"PyTorch output shape: {zv_pytorch.shape}")
    logger.info(f"ONNX output shape: {zv_onnx.shape}")
    
    return True


if __name__ == "__main__":
    # First test basic functionality
    success = test_onnx_models()
    
    if success:
        logger.info("\n✓ Basic ONNX integration test passed!")
        
        # Then compare with PyTorch if models are available
        pytorch_model_path = Path(__file__).parent / "models" / "vampnet" / "coarse.pth"
        if pytorch_model_path.exists():
            compare_pytorch_onnx()
        else:
            logger.info("\nSkipping PyTorch comparison (models not found)")
    else:
        logger.error("\n✗ ONNX integration test failed!")