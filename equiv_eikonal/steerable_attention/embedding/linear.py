from flax import linen as nn
from equiv_eikonal.utils import torch_compatible_dense


class FFNEmbedding(nn.Module):
    num_in: int
    num_hidden: int
    num_out: int

    @nn.compact
    def __call__(self, x):
        x = torch_compatible_dense(
            in_features=self.num_in, out_features=self.num_hidden
        )(x)
        x = nn.gelu(x)
        x = torch_compatible_dense(
            in_features=self.num_hidden, out_features=self.num_out
        )(x)
        return x
