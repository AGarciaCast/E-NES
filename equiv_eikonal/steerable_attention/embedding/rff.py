import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np


class RFFNet(nn.Module):
    in_dim: int
    output_dim: int
    hidden_dim: int
    num_layers: int
    learnable_coefficients: bool
    std: float
    learnable_std: bool = False
    numerator: float = 2.0
    norm: bool = False

    @nn.compact
    def __call__(self, x):
        assert (
            self.num_layers >= 2
        ), "At least two layers (the hidden plus the output one) are required."

        # Encoding via RFFEmbedding.
        x = RFFEmbedding(
            num_in=self.in_dim,
            num_hidden=self.hidden_dim,
            learnable_coefficients=self.learnable_coefficients,
            std=self.std,
            learnable_std=self.learnable_std,
            norm=self.norm,
            name="encoding",
        )(x)

        # Hidden layers.
        for i in range(self.num_layers - 1):
            x = Layer(
                hidden_dim=self.hidden_dim, numerator=self.numerator, name=f"layer_{i}"
            )(x)

        # Output layer.
        x = nn.Dense(
            features=self.output_dim,
            use_bias=True,
            kernel_init=nn.initializers.variance_scaling(
                self.numerator, "fan_in", "uniform"
            ),
            bias_init=nn.initializers.normal(stddev=1e-6),
            name="linear_final",
        )(x)

        return x


class RFFEmbedding(nn.Module):
    """
    Random Fourier Features (RFF) embedding module.

    Attributes:
        num_in: Number of input features.
        num_hidden: Number of hidden features (must be even).
        std: Initial value for the standard deviation parameter.
        learnable_coefficients: Whether coefficients are learnable.
        learnable_std: Whether the standard deviation is learnable.
        norm: Whether to apply normalization via AdaptiveSigmoid.
    """

    num_in: int
    num_hidden: int
    std: float  # Initial value for std_param
    learnable_coefficients: bool
    learnable_std: bool = False
    norm: bool = False

    def setup(self):
        # Ensure an even number of hidden features.
        assert self.num_hidden % 2 == 0, "num_hidden must be an even number"
        self.pi = 2 * jnp.pi

        # Make std either a trainable parameter or a fixed constant
        if self.learnable_std:
            self.std_param = self.param(
                "std_param",
                lambda _: jnp.asarray(self.std * jnp.ones((1,)), dtype=jnp.float32),
            )
            std = 1.0
        else:
            std = self.std

        # Initialize coefficients with unscaled normal distribution
        coeff_shape = (self.num_in, self.num_hidden // 2)
        if self.learnable_coefficients:
            self.coefficients = self.param(
                "coefficients",
                lambda rng, shape: std
                * jax.random.normal(rng, shape, dtype=jnp.float32),
                coeff_shape,
            )
        else:
            # Create a new PRNG key for coefficient initialization
            static_rng = jax.random.PRNGKey(0)
            self.coefficients = self.variable(
                "constants",
                "coefficients",
                lambda: std
                * jax.random.normal(static_rng, coeff_shape, dtype=jnp.float32),
            )

        if self.norm:
            self.act = AdaptiveSigmoid(self.num_in)

    def __call__(self, x):
        # Use the stored coefficients (non-trainable if applicable).
        coeffs = (
            self.coefficients.value
            if not self.learnable_coefficients
            else self.coefficients
        )

        # Apply scaling by std_param (access value if not learnable)
        if self.learnable_std:
            coeffs = self.std_param * coeffs

        if self.norm:
            x = self.act(x)
        x_proj = self.pi * jnp.matmul(x, coeffs)
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


class Layer(nn.Module):
    hidden_dim: int
    numerator: float = 2.0

    def setup(self):
        self.linear = nn.Dense(
            features=self.hidden_dim,
            use_bias=True,
            kernel_init=nn.initializers.variance_scaling(
                self.numerator, "fan_in", "normal"
            ),
            bias_init=nn.initializers.normal(stddev=1e-6),
        )
        self.activation = nn.relu

    def __call__(self, x):
        return self.activation(self.linear(x))


class AdaptiveSigmoid(nn.Module):
    dim: int

    def setup(self):
        self.scaling_factors = self.param(
            "scaling_factors", nn.initializers.ones, (self.dim,)
        )

    def __call__(self, x):
        scaled_x = x * self.scaling_factors
        return nn.sigmoid(scaled_x)


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    np.random.seed(0)

    # Test with fixed coefficients and fixed std
    print("\n=== Testing with fixed coefficients and fixed std ===")
    model = RFFEmbedding(
        num_in=3,
        num_hidden=4,
        std=1.0,
        learnable_coefficients=False,
        learnable_std=False,
    )
    variables = model.init(key, jnp.ones((1, 3)))

    initial_coeffs = variables["constants"]["coefficients"]
    initial_std = variables["constants"]["std_param"]
    print("Initial coefficients:", initial_coeffs)
    print("Initial std:", initial_std)

    for i in range(10):
        outputs = model.apply(variables, jnp.ones((1, 3)), mutable=["constants"])
        new_constants = outputs[1]["constants"]
        new_coeffs = new_constants["coefficients"]
        new_std = new_constants["std_param"]

        print(f"Coefficients at call {i+1}:", new_coeffs)
        print(f"Std at call {i+1}:", new_std)

        assert jnp.array_equal(
            initial_coeffs, new_coeffs
        ), f"Coefficients changed at call {i+1}, but they should remain the same"
        assert jnp.array_equal(
            initial_std, new_std
        ), f"Std changed at call {i+1}, but it should remain the same"

    print("Test passed: Coefficients and std do not change across 10 calls")

    # Test with learnable std
    print("\n=== Testing with learnable std ===")
    model_learnable = RFFEmbedding(
        num_in=3,
        num_hidden=4,
        std=1.0,
        learnable_coefficients=False,
        learnable_std=True,
    )
    variables_learnable = model_learnable.init(key, jnp.ones((1, 3)))

    print("Initial std (learnable):", variables_learnable["params"]["std_param"])

    # In a real training scenario, the optimizer would update std_param
    print("In training, std_param would be updated by the optimizer")
