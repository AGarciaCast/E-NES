"""
Equivariant Neural Field (ENF) model for solving the Eikonal equation.

This module implements the main Equivariant Neural Field architecture that combines
equivariant cross-attention mechanisms with dense neural networks to solve the
Eikonal equation while respecting geometric symmetries.
"""

from typing import Tuple, Any


from jax.nn import gelu
from flax import linen as nn

# Import modules
from equiv_eikonal.steerable_attention.equivariant_cross_attention import (
    EquivariantCrossAttention,
    PointwiseFFN,
)
from equiv_eikonal.models.dense import DenseBody

from equiv_eikonal.utils import torch_compatible_dense


class EquivariantNeuralField(nn.Module):
    """
    Equivariant Neural Field using cross-attention.

    This network processes spatial inputs conditioned on latent codes using equivariant
    cross-attention mechanisms.

    The architecture consists of:
    1. Latent code projection to hidden space
    2. Equivariant cross-attention layer with layer normalization
    3. Pointwise feed-forward network
    4. Dense output projection network

    Attributes:
        num_hidden: Number of hidden units.
        num_heads: Number of attention heads.
        latent_dim: Dimensionality of the input latent code.
        num_out: Number of output dimensions (typically 1 for travel time).
        invariant: Instance of BaseThreewayInvariants for computing geometric invariants.
        embedding_type: Type of positional embedding ('rff' for Random Fourier Features).
        embedding_freq_multiplier: Tuple of (min, max) frequency multipliers for embeddings.
    """

    num_hidden: int
    num_heads: int
    latent_dim: int
    num_out: int
    invariant: Any
    embedding_type: str
    embedding_freq_multiplier: Tuple[float, float]

    def setup(self):
        """
        Initialize all sub-modules of the Equivariant Neural Field.

        Sets up the latent stem projection, layer normalization, equivariant attention,
        feed-forward network, and output projection layers.
        """
        self.activation = gelu

        # Map latent code to hidden space.
        self.latent_stem = torch_compatible_dense(
            in_features=self.latent_dim, out_features=self.num_hidden
        )

        # Pre-norm layer for attention input (helps training stability)
        # self.layer_norm_stem = nn.LayerNorm()
        self.layer_norm_attn = nn.LayerNorm(
            epsilon=1e-5,
            use_fast_variance=False,
            force_float32_reductions=False,
        )
        # self.layer_norm_ffn = nn.LayerNorm()

        # Equivariant cross attention module - core of the architecture
        self.attn = EquivariantCrossAttention(
            num_hidden=self.num_hidden,
            num_heads=self.num_heads,
            invariant=self.invariant,
            embedding_type=self.embedding_type,
            embedding_freq_multiplier=self.embedding_freq_multiplier,
        )

        # Pointwise feed-forward block for post-attention processing
        self.pointwise_ffn = PointwiseFFN(
            num_in=self.num_heads * self.num_hidden,
            num_hidden=self.num_heads * self.num_hidden,
            num_out=self.num_heads * self.num_hidden,
        )

        # Output projection block - maps to final scalar output (travel time)
        self.out_proj = DenseBody(
            input_dim=self.num_heads * self.num_hidden,
            nu=self.num_hidden,
            nl=3,
            out_dim=1,
            act="ad-gauss-1",  # Adaptive Gaussian activation
            # act="gelu",
            out_act="linear",
        )

    def __call__(self, inputs, p, a):
        """
        Forward pass through the Equivariant Neural Field.
        """
        # Map latent features to hidden dimension
        a = self.latent_stem(a)

        # Pre-norm residual attention block - applies equivariant cross-attention
        out = self.attn(inputs, p, self.layer_norm_attn(a))

        # Pre-norm residual feed-forward block - processes attention output
        out = self.pointwise_ffn(out)
        out = self.activation(out)

        # Final projection to output dimension
        out = self.out_proj(out)
        return out
