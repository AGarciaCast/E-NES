from typing import Tuple, Any
import jax.numpy as jnp
from jax.nn import gelu
from flax import linen as nn


# Query embedding
from equiv_eikonal.steerable_attention.embedding import get_embedding


from equiv_eikonal.utils import (
    torch_compatible_dense,
    stable_softmax,
)


class PointwiseFFN(nn.Module):
    num_in: int
    num_hidden: int
    num_out: int

    @nn.compact
    def __call__(self, x):
        x = torch_compatible_dense(self.num_in, self.num_hidden)(x)
        x = gelu(x)
        x = nn.LayerNorm(
            epsilon=1e-5,
            use_fast_variance=False,
            force_float32_reductions=False,
        )(x)
        x = torch_compatible_dense(self.num_hidden, self.num_out)(x)
        return x


class EquivariantCrossAttention(nn.Module):
    num_hidden: int
    num_heads: int
    invariant: Any  # "BaseThreewayInvariants"
    embedding_type: str
    embedding_freq_multiplier: Tuple[float, float]

    def setup(self):
        self.symmetric = self.invariant.symmetric
        self.scale = jnp.asarray(1.0 / jnp.sqrt(self.num_hidden), dtype=jnp.float32)

        emb_mult_inv, emb_mult_val = self.embedding_freq_multiplier
        self.invariant_embedding_query = get_embedding(
            embedding_type=self.embedding_type,
            num_in=self.invariant.dim,
            num_hidden=self.num_hidden,
            num_emb_dim=self.num_hidden,
            freq_multiplier=emb_mult_inv,
        )
        self.invariant_embedding_value = get_embedding(
            embedding_type=self.embedding_type,
            num_in=self.invariant.dim,
            num_hidden=self.num_hidden,
            num_emb_dim=self.num_hidden,
            freq_multiplier=emb_mult_val,
        )

        self.inv_emb_to_q = torch_compatible_dense(
            self.num_hidden, self.num_heads * self.num_hidden
        )
        self.a_to_k = torch_compatible_dense(
            self.num_hidden, self.num_heads * self.num_hidden
        )
        self.a_to_v = torch_compatible_dense(
            self.num_hidden, self.num_heads * self.num_hidden
        )

        self.inv_emb_to_v = PointwiseFFN(
            num_in=self.num_hidden,
            num_hidden=self.num_hidden,
            num_out=2 * self.num_heads * self.num_hidden,
        )

        self.inv_emb_cond_mixer = PointwiseFFN(
            num_in=self.num_heads * self.num_hidden,
            num_hidden=self.num_heads * self.num_hidden,
            num_out=self.num_heads * self.num_hidden,
        )

        self.out_proj = torch_compatible_dense(
            self.num_heads * self.num_hidden, self.num_heads * self.num_hidden
        )

    def __call__(self, inputs, p, a):
        inv = self.invariant(inputs, p)
        if self.symmetric:
            inv_emb_q_xy = self.invariant_embedding_query(inv[0])
            inv_emb_q_yx = self.invariant_embedding_query(inv[1])
            inv_emb_q = (inv_emb_q_xy + inv_emb_q_yx) * 0.5

            inv_emb_v_xy = self.invariant_embedding_value(inv[0])
            inv_emb_v_yx = self.invariant_embedding_value(inv[1])
            inv_emb_v = (inv_emb_v_xy + inv_emb_v_yx) * 0.5
        else:
            inv_emb_q = self.invariant_embedding_query(inv)
            inv_emb_v = self.invariant_embedding_value(inv)

        q = self.inv_emb_to_q(inv_emb_q)
        k = self.a_to_k(a)
        v = self.a_to_v(a)

        v_gamma_beta = self.inv_emb_to_v(inv_emb_v)
        v_gamma, v_beta = jnp.split(v_gamma_beta, 2, axis=-1)
        v = v[:, None, :, :] * (1 + v_gamma) + v_beta
        v = self.inv_emb_cond_mixer(v)

        v = v.reshape(v.shape[:-1] + (self.num_heads, self.num_hidden))
        q = q.reshape(q.shape[:-1] + (self.num_heads, self.num_hidden))
        k = k.reshape(k.shape[:-1] + (self.num_heads, self.num_hidden))

        # Compute attention with improved numerical stability
        att_logits = jnp.einsum("bczhd,bzhd->bczh", q, k) * self.scale

        # Apply stable softmax
        att = stable_softmax(att_logits, axis=-2)

        # Apply attention weights with stable calculation
        y = jnp.einsum("bczh,bczhd->bchd", att, v)
        y = y.reshape(*y.shape[:2], self.num_heads * self.num_hidden)
        y = self.out_proj(y)
        return y
