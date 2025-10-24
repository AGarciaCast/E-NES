from typing import Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn

from equiv_eikonal.models.dense import DenseBody

class MLP(nn.Module):
    hidden_sizes: Tuple[int, int]
    num_out: int

    @nn.compact
    def __call__(self, c):
        for hidden_size in self.hidden_sizes:
            c = nn.Dense(hidden_size)(c)
            c = nn.relu(c)
        c = nn.Dense(self.num_out)(c)
        return c

class SineLayer(nn.Module):
    num_in: int
    num_hidden: int
    w0: float
    is_first: bool = False

    def setup(self):
        # Calc init range
        init_range = 1 / self.num_in if self.is_first else jnp.sqrt(6 / self.num_in) / self.w0
        self.linear = nn.Dense(self.num_hidden, kernel_init=nn.initializers.uniform(-init_range, init_range))

    def __call__(self, x, mod):
        # Init dense layer with init range
        x_proj = self.linear(x)

        # Split modulation into beta gamma
        beta, gamma = jnp.split(mod, 2, axis=-1)

        # FiLM modulation
        x_proj = beta[:, None, :] * x_proj + gamma[:, None, :]
        return jnp.sin(self.w0 * x_proj)

class LatentModulatedSiren(nn.Module):
    num_in: int
    num_hidden: int
    num_layers: int
    num_out: int
    modulation_hidden_sizes: Tuple[int, int]
    w0: float

    def setup(self):
        # Calculate the number of modulation output units. 2x the number of hidden units x the number of layers.
        num_modulation_out = 2 * self.num_hidden * self.num_layers
        self.mod_mlp = MLP(self.modulation_hidden_sizes, num_modulation_out)

        # Create the SineLayer modules
        layers = []
        for i in range(self.num_layers):
            layers.append(SineLayer(self.num_in if i == 0 else self.num_hidden, self.num_hidden, self.w0, is_first=(i == 0)))
        self.layers = layers

        # Create the final output layer
        self.out = nn.Dense(self.num_out)

        self.out1 = DenseBody(
            input_dim= self.num_hidden,
            nu=self.num_hidden,
            nl=3,
            out_dim=1,
            act="ad-gauss-1",
            # act="gelu",
            out_act="linear",
        )

    def __call__(self, x, p, c):
        # Reshape coordinate pairs to single vector for FUNCTA
        x = x.reshape(x.shape[0], x.shape[1], -1)

        # Map c to modulation parameters
        mod = self.mod_mlp(c)

        # Split per layer
        mod_per_layer = jnp.split(mod, self.num_layers, axis=-1)

        # Apply the layers
        for i, layer in enumerate(self.layers):
            x = layer(x, mod_per_layer[i])

        # Apply the final output layer
        x = self.out(x) + 0.5
        x= nn.relu(x)
        return self.out1(x) 


if __name__ == "__main__":
    # Test the Siren module
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    x = jax.random.uniform(subkey, (64, 256, 2))

    # Initialize the model
    model = LatentModulatedSiren(num_in=2, num_hidden=256, num_layers=12, num_out=1, modulation_hidden_sizes=(256, 512), w0=30)
    params = model.init(subkey, x, jnp.zeros((1, 256)))

    # Perform a forward pass
    y = model.apply(params, x, jnp.zeros((1, 256)))
    print(y)
    print(y.shape)
    # Expected output: (1, 1)
