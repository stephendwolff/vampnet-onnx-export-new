import time
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def recurse_children(module, fn):
    for child in module.children():
        if isinstance(child, nn.ModuleList):
            for c in child:
                yield recurse_children(c, fn)
        if isinstance(child, nn.ModuleDict):
            for c in child.values():
                yield recurse_children(c, fn)

        yield recurse_children(child, fn)
        yield fn(child)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class SequentialWithFiLM(nn.Module):
    """
    handy wrapper for nn.Sequential that allows FiLM layers to be
    inserted in between other layers.
    """

    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    @staticmethod
    def has_film(module):
        mod_has_film = any(
            [res for res in recurse_children(module, lambda c: isinstance(c, FiLM))]
        )
        return mod_has_film

    def forward(self, x, cond):
        for layer in self.layers:
            if self.has_film(layer):
                x = layer(x, cond)
            else:
                x = layer(x)
        return x


class FiLM(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if input_dim > 0:
            self.beta = nn.Linear(input_dim, output_dim)
            self.gamma = nn.Linear(input_dim, output_dim)

    def forward(self, x, r):
        if self.input_dim == 0:
            return x
        else:
            beta, gamma = self.beta(r), self.gamma(r)
            beta, gamma = (
                beta.view(x.size(0), self.output_dim, 1),
                gamma.view(x.size(0), self.output_dim, 1),
            )
            x = x * (gamma + 1) + beta
        return x


class CodebookEmbedding(nn.Module):
    """
    ONNX-compatible CodebookEmbedding that uses pre-saved codebook lookup tables
    instead of accessing codec.quantizer.quantizers[i].codebook.weight
    """

    def __init__(
        self,
        vocab_size: int,
        latent_dim: int,
        n_codebooks: int,
        emb_dim: int,
        lookup_tables_path: str,
        special_tokens: Optional[Tuple[str]] = None,
    ):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.emb_dim = emb_dim
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size

        # Load pre-saved codebook lookup tables
        self.lookup_tables = torch.load(lookup_tables_path, map_location="cpu")
        print(f"✓ Loaded {len(self.lookup_tables)} codebook lookup tables")

        # Validate lookup tables
        # assert (
        #     len(self.lookup_tables) == n_codebooks
        # ), f"Expected {n_codebooks} codebooks, got {len(self.lookup_tables)}"

        # Convert to nn.Parameters for proper device handling
        self.codebook_embeddings = nn.ParameterList(
            [
                nn.Parameter(table.clone(), requires_grad=False)
                for table in self.lookup_tables
            ]
        )

        # Handle special tokens
        self.special_tokens = special_tokens
        if special_tokens is not None:
            self.special = nn.ParameterDict(
                {
                    tkn: nn.Parameter(torch.randn(n_codebooks, self.latent_dim))
                    for tkn in special_tokens
                }
            )
            self.special_idxs = {
                tkn: i + vocab_size for i, tkn in enumerate(special_tokens)
            }
        else:
            self.special = None
            self.special_idxs = {}

        self.out_proj = nn.Conv1d(n_codebooks * self.latent_dim, self.emb_dim, 1)

    def get_lookup_table(self, codebook_idx: int) -> torch.Tensor:
        """
        Get lookup table for a specific codebook index

        Args:
            codebook_idx: Index of the codebook (0 to n_codebooks-1)

        Returns:
            Lookup table tensor of shape (vocab_size, latent_dim)
        """
        base_table = self.codebook_embeddings[codebook_idx]

        if self.special is not None:
            # Add special token embeddings
            special_lookup = torch.stack(
                [self.special[tkn][codebook_idx] for tkn in self.special_tokens], dim=0
            )
            lookup_table = torch.cat([base_table, special_lookup], dim=0)
        else:
            lookup_table = base_table

        return lookup_table

    def from_codes(self, codes: torch.Tensor, codec=None) -> torch.Tensor:
        """
        Get continuous embeddings from discrete codes (ONNX-compatible version)

        Args:
            codes: Discrete codes tensor of shape (batch, n_codebooks, time)
            codec: Ignored (kept for compatibility with original interface)

        Returns:
            Latent embeddings of shape (batch, n_codebooks * latent_dim, time)
        """
        n_codebooks = codes.shape[1]
        latent = []

        for i in range(n_codebooks):
            c = codes[:, i, :]  # Shape: (batch, time)

            # Get lookup table for this codebook
            lookup_table = self.get_lookup_table(i)

            # Embedding lookup
            l = F.embedding(c, lookup_table).transpose(
                1, 2
            )  # (batch, latent_dim, time)
            latent.append(l)

        # Concatenate all codebook embeddings
        latent = torch.cat(latent, dim=1)  # (batch, n_codebooks * latent_dim, time)
        return latent

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Project a sequence of latents to a sequence of embeddings

        Args:
            latents: Input latents of shape (batch, n_codebooks * latent_dim, time)

        Returns:
            Projected embeddings of shape (batch, emb_dim, time)
        """
        x = self.out_proj(latents)
        return x
