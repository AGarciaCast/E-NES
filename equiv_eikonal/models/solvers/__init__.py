from equiv_eikonal.models.solvers.euclidean_solver import (
    EuclidenEquivariantNeuralEikonalSolver,
)

from equiv_eikonal.models.solvers.position_orientation_solver import (
    PositionOrientationEquivariantNeuralEikonalSolver,
)

from equiv_eikonal.models.solvers.sphere_solver import (
    SphereEquivariantNeuralEikonalSolver,
)


import jax.numpy as jnp

# Import modules
from equiv_eikonal.steerable_attention.threeway_invariants import BaseThreewayInvariants


def get_solver(
    cfg,
    invariant: BaseThreewayInvariants,
    vmin: float = 0.0,
    vmax: float = 1.0,
):
    space = cfg.geometry.input_space
    num_hidden = cfg.solver.num_hidden
    num_heads = cfg.solver.num_heads
    latent_dim = cfg.geometry.latent_dim
    embedding_type = cfg.solver.embedding_type
    embedding_freq_multiplier = [
        cfg.solver.embedding_freq_multiplier_invariant,
        cfg.solver.embedding_freq_multiplier_value,
    ]
    factored = cfg.eikonal.factored

    if space == "Euclidean":
        return EuclidenEquivariantNeuralEikonalSolver(
            num_hidden=num_hidden,
            num_heads=num_heads,
            latent_dim=latent_dim,
            invariant=invariant,
            embedding_type=embedding_type,
            embedding_freq_multiplier=embedding_freq_multiplier,
            vmin=vmin,
            vmax=vmax,
            factored=factored,
        )
    elif space == "Position_Orientation":
        return PositionOrientationEquivariantNeuralEikonalSolver(
            num_hidden=num_hidden,
            num_heads=num_heads,
            latent_dim=latent_dim,
            invariant=invariant,
            embedding_type=embedding_type,
            embedding_freq_multiplier=embedding_freq_multiplier,
            vmin=vmin,
            vmax=vmax,
            factored=factored,
            xi=cfg.geometry.metric.xi,
            epsilon=cfg.geometry.metric.epsilon,
            theta_range=(
                [
                    0.0,
                    2.0 * jnp.pi,
                ]
                if cfg.geometry.theta_range == "zero"
                else [-jnp.pi, jnp.pi]
            ),
        )
    elif space == "Spherical":
        return SphereEquivariantNeuralEikonalSolver(
            num_hidden=num_hidden,
            num_heads=num_heads,
            latent_dim=latent_dim,
            invariant=invariant,
            embedding_type=embedding_type,
            embedding_freq_multiplier=embedding_freq_multiplier,
            vmin=vmin,
            vmax=vmax,
            factored=factored,
            distance_type=cfg.geometry.distance_type,
        )
    else:
        raise ValueError()
