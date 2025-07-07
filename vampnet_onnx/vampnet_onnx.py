import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Optional, Dict, Tuple
import logging

from .util import codebook_flatten, codebook_unflatten, scalar_to_batch_tensor
from .mask import _gamma


class VampNetONNX(nn.Module):
    """ONNX-based VampNet model for inference"""
    
    def __init__(
        self,
        embeddings_path: str,
        transformer_path: str,
        n_codebooks: int,
        n_conditioning_codebooks: int,
        n_predict_codebooks: int,
        vocab_size: int,
        mask_token: int,
        device: str = "cpu"
    ):
        super().__init__()
        self.embeddings_path = Path(embeddings_path)
        self.transformer_path = Path(transformer_path)
        self.n_codebooks = n_codebooks
        self.n_conditioning_codebooks = n_conditioning_codebooks
        self.n_predict_codebooks = n_predict_codebooks
        self.vocab_size = vocab_size
        self.mask_token = mask_token
        self.device = device
        
        # Initialize ONNX sessions
        providers = ['CPUExecutionProvider']
        if device == "cuda":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
        self.embed_session = ort.InferenceSession(str(self.embeddings_path), providers=providers)
        self.trans_session = ort.InferenceSession(str(self.transformer_path), providers=providers)
        
        logging.info(f"Loaded ONNX models from {self.embeddings_path} and {self.transformer_path}")
        
    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        Args:
            codes: (batch, n_codebooks, time)
        Returns:
            logits: (batch, n_predict_codebooks * vocab_size, time)
        """
        # Convert to numpy for ONNX
        codes_np = codes.cpu().numpy()
        
        # Get embeddings
        latents = self.embed_session.run(None, {"codes": codes_np})[0]
        
        # Run through transformer
        logits = self.trans_session.run(None, {"latents": latents})[0]
        
        # Convert back to torch
        logits = torch.from_numpy(logits).to(self.device)
        
        # Logits from ONNX are in the format: (batch, vocab_size, time * n_predict_codebooks)
        # This matches the PyTorch model's output format after rearrange
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        codec,
        time_steps: int = 300,
        _sampling_steps: int = 12,
        sampling_steps: int = None,  # alias for _sampling_steps
        start_tokens: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        mask: Optional[torch.Tensor] = None,
        mask_temperature: float = 10.5,
        ctrls: dict = None,
        ctrl_masks: dict = None,
        typical_filtering=True,
        typical_mass=0.15,
        typical_min_tokens=64,
        top_p=None,
        seed: int = None,
        sample_cutoff: float = 1.0,
        return_signal=True,
        debug=False,
        causal_weight: float = 0.0,
        cfg_scale: float = 3.0,
        cfg_guidance: float = None,
        cond=None,  # unused
        **kwargs,  # catch any other arguments
    ):
        """Generate function compatible with PyTorch VampNet interface"""
        # Handle both sampling_steps and _sampling_steps
        if sampling_steps is not None:
            _sampling_steps = sampling_steps
            
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        sampling_steps = _sampling_steps
        logging.debug(f"beginning generation with {sampling_steps} steps")
        
        # Initialize z
        z = start_tokens
        if z is None:
            z = torch.full((1, self.n_codebooks, time_steps), self.mask_token).to(self.device)
            
        # Initialize mask
        if mask is None:
            mask = torch.ones_like(z).to(self.device).int()
            mask[:, :self.n_conditioning_codebooks, :] = 0.0
        if mask.ndim == 2:
            mask = mask[:, None, :].repeat(1, z.shape[1], 1)
            
        # Apply initial mask
        z_masked = z.masked_fill(mask.bool(), self.mask_token)
        
        # Keep track of the original mask - tokens that should never be changed
        original_mask = mask.clone()
        
        logging.debug(f"Generate: z shape: {z.shape}, n_codebooks: {self.n_codebooks}, n_predict: {self.n_predict_codebooks}")
        
        # Count mask tokens
        num_mask_tokens_at_start = (z_masked == self.mask_token).sum()
        
        # Sampling loop
        for i in range(sampling_steps):
            # Schedule parameter
            r = scalar_to_batch_tensor((i + 1) / sampling_steps, z.shape[0]).to(z.device)
            
            # Forward pass
            logits = self.forward(z_masked)  # (batch, vocab_size, time * n_codebooks)
            
            # Reshape logits for sampling
            # The model outputs in the format used by VampNet after rearrange
            batch_size = logits.shape[0]
            vocab_size = logits.shape[1]
            seq_len = logits.shape[2]  # This is time * n_codebooks
            
            # First permute to (batch, seq, vocab) for sampling
            logits = logits.permute(0, 2, 1)  # (batch, time * n_codebooks, vocab_size)
            
            # For sampling, we need to handle multiple codebooks
            if self.n_predict_codebooks > 1:
                # The sequence is already time * n_codebooks, need to separate them
                # From (batch, time * n_codebooks, vocab) to (batch, time, n_codebooks, vocab)
                logits_reshaped = logits.reshape(batch_size, time_steps, self.n_predict_codebooks, vocab_size)
                # Sample for each codebook position
                sampled_tokens = []
                sampled_probs = []
                for c in range(self.n_predict_codebooks):
                    logits_c = logits_reshaped[:, :, c, :].reshape(-1, self.vocab_size)
                    tokens_c, probs_c = sample_from_logits(
                        logits_c,
                        sample=((i / sampling_steps) <= sample_cutoff),
                        temperature=temperature,
                        typical_filtering=typical_filtering,
                        typical_mass=typical_mass,
                        typical_min_tokens=typical_min_tokens,
                        top_k=None,
                        top_p=top_p,
                        return_probs=True,
                    )
                    sampled_tokens.append(tokens_c.reshape(batch_size, time_steps))
                    sampled_probs.append(probs_c.reshape(batch_size, time_steps))
                
                # Stack codebooks: (batch, n_predict, time)
                sampled_z = torch.stack(sampled_tokens, dim=1)
                selected_probs = torch.stack(sampled_probs, dim=1)
                # Flatten for processing: (batch, time * n_predict)
                sampled_z = codebook_flatten(sampled_z)
                selected_probs = codebook_flatten(selected_probs)
            else:
                # Single codebook
                logits_for_sampling = logits.reshape(-1, self.vocab_size)
                sampled_z, selected_probs = sample_from_logits(
                    logits_for_sampling,
                    sample=((i / sampling_steps) <= sample_cutoff),
                    temperature=temperature,
                    typical_filtering=typical_filtering,
                    typical_mass=typical_mass,
                    typical_min_tokens=typical_min_tokens,
                    top_k=None,
                    top_p=top_p,
                    return_probs=True,
                )
            
            logging.debug(f"Step {i}: sampled_z shape: {sampled_z.shape}, logits shape: {logits.shape}")
            
            # For coarse model, sampled_z is already the right shape
            # It's flattened across codebooks: (batch, time * n_predict_codebooks)
                
            # Handle codebook dimension
            if self.n_conditioning_codebooks > 0:
                # Flatten only the non-conditioning part for processing
                z_masked_flat = codebook_flatten(z_masked[:, self.n_conditioning_codebooks:, :])
                z_flat = codebook_flatten(z[:, self.n_conditioning_codebooks:, :])
                original_mask_flat = codebook_flatten(original_mask[:, self.n_conditioning_codebooks:, :])
                mask_flat = (z_masked_flat == self.mask_token).int()
                
                # Update with sampled values only where currently masked
                sampled_z = torch.where(mask_flat.bool(), sampled_z, z_masked_flat)
                # For unmasked positions (original_mask == 0), keep original values
                sampled_z = torch.where(~original_mask_flat.bool(), z_flat, sampled_z)
                selected_probs = torch.where(mask_flat.bool(), selected_probs, torch.inf)
                
                # Calculate number to mask
                num_to_mask = torch.floor(_gamma(r) * num_mask_tokens_at_start).unsqueeze(1).long()
                
                if i != (sampling_steps - 1):
                    num_to_mask = torch.maximum(
                        torch.tensor(1),
                        torch.minimum(mask_flat.sum(dim=-1, keepdim=True) - 1, num_to_mask),
                    )
                
                # Get new mask
                mask_flat = mask_by_random_topk(num_to_mask, selected_probs, mask_temperature * (1 - r))
                
                # Update z_masked
                z_masked_flat = torch.where(mask_flat.bool(), self.mask_token, sampled_z)
                
                # Ensure original unmasked tokens remain unchanged
                z_masked_flat = torch.where(~original_mask_flat.bool(), z_flat, z_masked_flat)
                
                # Unflatten
                z_masked_unflat = codebook_unflatten(z_masked_flat, self.n_predict_codebooks)
                
                # Add conditioning codebooks back
                z_masked = torch.cat(
                    (z[:, :self.n_conditioning_codebooks, :], z_masked_unflat), dim=1
                )
            else:
                # No conditioning codebooks (coarse model)
                # Flatten z_masked for processing
                z_masked_flat = codebook_flatten(z_masked)
                z_flat = codebook_flatten(z)
                original_mask_flat = codebook_flatten(original_mask)
                mask_flat = (z_masked_flat == self.mask_token).int()
                
                # Update with sampled values only where currently masked
                sampled_z = torch.where(mask_flat.bool(), sampled_z, z_masked_flat)
                # For unmasked positions (original_mask == 0), keep original values
                sampled_z = torch.where(~original_mask_flat.bool(), z_flat, sampled_z)
                selected_probs = torch.where(mask_flat.bool(), selected_probs, torch.inf)
                
                # Calculate number to mask
                num_to_mask = torch.floor(_gamma(r) * num_mask_tokens_at_start).unsqueeze(1).long()
                
                if i != (sampling_steps - 1):
                    num_to_mask = torch.maximum(
                        torch.tensor(1),
                        torch.minimum(mask_flat.sum(dim=-1, keepdim=True) - 1, num_to_mask),
                    )
                
                # Get new mask
                mask_flat = mask_by_random_topk(num_to_mask, selected_probs, mask_temperature * (1 - r))
                
                # Update z_masked
                z_masked_flat = torch.where(mask_flat.bool(), self.mask_token, sampled_z)
                
                # Ensure original unmasked tokens remain unchanged
                z_masked_flat = torch.where(~original_mask_flat.bool(), z_flat, z_masked_flat)
                
                # Unflatten
                z_masked = codebook_unflatten(z_masked_flat, self.n_codebooks)
        
        # Final step: ensure original unmasked tokens are preserved in the output
        z_masked = torch.where(~original_mask.bool(), z, z_masked)
        
        if return_signal:
            return self.decode(z_masked, codec)
        else:
            return z_masked
    
    def decode(self, z: torch.Tensor, codec):
        """Decode tokens to audio signal"""
        # Check if there are any mask tokens - this should not happen
        if (z == self.mask_token).any():
            import logging
            mask_count = (z == self.mask_token).sum().item()
            total_tokens = z.numel()
            logging.warning(f"Found {mask_count}/{total_tokens} mask tokens in decode input! This should not happen.")
            
            # Debug info
            for cb in range(z.shape[1]):
                cb_mask_count = (z[:, cb, :] == self.mask_token).sum().item()
                if cb_mask_count > 0:
                    logging.warning(f"  Codebook {cb}: {cb_mask_count} mask tokens")
            
            # For now, just replace with 0 to match original behavior
            # but log a warning
            z = z.masked_fill(z == self.mask_token, 0)
        
        # Use codec to decode
        signal = codec.decode_from_codes(z)["audio"]
        
        # Return as AudioSignal
        from audiotools import AudioSignal
        return AudioSignal(signal, codec.sample_rate)


class VampNetCoarseONNX(VampNetONNX):
    """Coarse VampNet model using ONNX"""
    
    def __init__(
        self,
        embeddings_path: str = "vampnet_coarse_embeddings.onnx",
        transformer_path: str = "vampnet_coarse_transformer.onnx",
        n_codebooks: int = 4,
        vocab_size: int = 1024,
        mask_token: int = 1024,
        device: str = "cpu"
    ):
        super().__init__(
            embeddings_path=embeddings_path,
            transformer_path=transformer_path,
            n_codebooks=n_codebooks,
            n_conditioning_codebooks=0,
            n_predict_codebooks=n_codebooks,
            vocab_size=vocab_size,
            mask_token=mask_token,
            device=device
        )


class VampNetC2FONNX(VampNetONNX):
    """Coarse-to-fine VampNet model using ONNX"""
    
    def __init__(
        self,
        embeddings_path: str = "vampnet_c2f_embeddings.onnx",
        transformer_path: str = "vampnet_c2f_transformer.onnx",
        n_codebooks: int = 14,
        n_conditioning_codebooks: int = 4,
        vocab_size: int = 1024,
        mask_token: int = 1024,
        device: str = "cpu"
    ):
        super().__init__(
            embeddings_path=embeddings_path,
            transformer_path=transformer_path,
            n_codebooks=n_codebooks,
            n_conditioning_codebooks=n_conditioning_codebooks,
            n_predict_codebooks=n_codebooks - n_conditioning_codebooks,
            vocab_size=vocab_size,
            mask_token=mask_token,
            device=device
        )


def sample_from_logits(
    logits,
    sample: bool = True,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    typical_filtering: bool = False,
    typical_mass: float = 0.2,
    typical_min_tokens: int = 1,
    return_probs: bool = False,
):
    """Sample from logits with various sampling strategies"""
    shp = logits.shape[:-1]
    
    if typical_filtering:
        # Apply typical filtering
        logits = typical_filter(logits, typical_mass=typical_mass, typical_min_tokens=typical_min_tokens)
    
    # Apply top_k sampling
    if top_k is not None:
        v, _ = logits.topk(top_k)
        logits[logits < v[..., [-1]]] = -float("inf")
    
    # Apply top_p (nucleus) sampling
    if top_p is not None and top_p < 1.0:
        v, sorted_indices = logits.sort(descending=True)
        cumulative_probs = v.softmax(dim=-1).cumsum(dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove = F.pad(sorted_indices_to_remove, (1, 0), value=False)[..., :-1]
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        
        logits[indices_to_remove] = -float("inf")
    
    # Perform multinomial sampling
    probs = F.softmax(logits / temperature, dim=-1) if temperature > 0 else logits.softmax(dim=-1)
    token = (
        probs.view(-1, probs.size(-1)).multinomial(1).squeeze(1).view(*shp)
        if sample
        else logits.argmax(-1)
    )
    
    if return_probs:
        token_probs = probs.take_along_dim(token.unsqueeze(-1), dim=-1).squeeze(-1)
        return token, token_probs
    else:
        return token


def mask_by_random_topk(num_to_mask: int, probs: torch.Tensor, temperature: float = 1.0):
    """Mask by random top-k selection"""
    noise = torch.zeros_like(probs).uniform_(1e-20, 1)
    noise = -torch.log(-torch.log(noise))
    
    temperature = temperature.unsqueeze(-1)
    confidence = torch.log(probs) + temperature * noise
    
    sorted_confidence, _ = confidence.sort(dim=-1)
    cut_off = torch.take_along_dim(sorted_confidence, num_to_mask, axis=-1)
    mask = confidence < cut_off
    
    return mask


def typical_filter(logits, typical_mass: float = 0.95, typical_min_tokens: int = 1):
    """Apply typical filtering to logits"""
    # Handle both 2D and 3D inputs
    if logits.ndim == 2:
        # Already flattened
        x_flat = logits
    else:
        # 3D tensor
        nb, nt, _ = logits.shape
        from einops import rearrange
        x_flat = rearrange(logits, "b t l -> (b t ) l")
    
    x_flat_norm = torch.nn.functional.log_softmax(x_flat, dim=-1)
    x_flat_norm_p = torch.exp(x_flat_norm)
    entropy = -(x_flat_norm * x_flat_norm_p).nansum(-1, keepdim=True)
    
    c_flat_shifted = torch.abs((-x_flat_norm) - entropy)
    c_flat_sorted, x_flat_indices = torch.sort(c_flat_shifted, descending=False)
    x_flat_cumsum = x_flat.gather(-1, x_flat_indices).softmax(dim=-1).cumsum(dim=-1)
    
    last_ind = (x_flat_cumsum < typical_mass).sum(dim=-1)
    sorted_indices_to_remove = c_flat_sorted > c_flat_sorted.gather(1, last_ind.view(-1, 1))
    
    if typical_min_tokens > 1:
        sorted_indices_to_remove[..., :typical_min_tokens] = 0
        
    indices_to_remove = sorted_indices_to_remove.scatter(1, x_flat_indices, sorted_indices_to_remove)
    x_flat = x_flat.masked_fill(indices_to_remove, -float("Inf"))
    
    # Return in the same shape as input
    if logits.ndim == 2:
        return x_flat
    else:
        from einops import rearrange
        return rearrange(x_flat, "(b t) l -> b t l", t=nt)