from equiv_eikonal.latents.vanilla_affine_orthogonal import *
from equiv_eikonal.latents.vanilla_orthogonal import *


def get_latents(num_signals, cfg, meta=False):

    geometry_cfg = cfg.geometry
    if geometry_cfg.input_space == "Euclidean":

        if geometry_cfg.triv == "vanilla":
            if not meta:
                return VanillaUncoupledAffineOrthogonalLatents(
                    num_signals=num_signals,
                    num_latents=geometry_cfg.num_latents,
                    dim_signal=geometry_cfg.dim_signal,
                    dim_orientation=geometry_cfg.dim_orientation,
                    latent_dim=geometry_cfg.latent_dim,
                    xmin=jnp.array(
                        (
                            cfg.data.x_min
                            if cfg.data.x_min is not None
                            else [-1.0] * geometry_cfg.dim_signal
                        ),
                        dtype=jnp.float32,
                    ),
                    xmax=jnp.array(
                        (
                            cfg.data.x_max
                            if cfg.data.x_max is not None
                            else [1.0] * geometry_cfg.dim_signal
                        ),
                        dtype=jnp.float32,
                    ),
                    init_pos=True,
                    norm_angles=False,
                )
            else:
                return VanillaMetaUncoupledAffineOrthogonalLatents(
                    num_signals=num_signals,
                    num_latents=geometry_cfg.num_latents,
                    dim_signal=geometry_cfg.dim_signal,
                    dim_orientation=geometry_cfg.dim_orientation,
                    latent_dim=geometry_cfg.latent_dim,
                    xmin=jnp.array(
                        (
                            cfg.data.x_min
                            if cfg.data.x_min is not None
                            else [-1.0] * geometry_cfg.dim_signal
                        ),
                        dtype=jnp.float32,
                    ),
                    xmax=jnp.array(
                        (
                            cfg.data.x_max
                            if cfg.data.x_max is not None
                            else [1.0] * geometry_cfg.dim_signal
                        ),
                        dtype=jnp.float32,
                    ),
                    init_pos=True,
                    norm_angles=False,
                )
        else:
            raise ValueError("There is no Riemannian version in Jax implementation")

    elif geometry_cfg.input_space == "Position_Orientation":

        if geometry_cfg.triv == "vanilla":
            if not meta:
                return VanillaPositionOrientationUncoupledAffineOrthogonalLatents(
                    num_signals=num_signals,
                    num_latents=geometry_cfg.num_latents,
                    dim_signal=2,
                    dim_orientation=geometry_cfg.dim_orientation,
                    latent_dim=geometry_cfg.latent_dim,
                    xmin=jnp.array(
                        cfg.data.x_min if cfg.data.x_min is not None else [-1.0, -1.0],
                        dtype=jnp.float32,
                    ),
                    xmax=jnp.array(
                        cfg.data.x_max if cfg.data.x_max is not None else [1.0, 1.0],
                        dtype=jnp.float32,
                    ),
                    init_pos=False,
                    norm_angles=False,
                )
            else:
                return VanillaMetaPositionOrientationUncoupledAffineOrthogonalLatents(
                    num_signals=num_signals,
                    num_latents=geometry_cfg.num_latents,
                    dim_signal=2,
                    dim_orientation=geometry_cfg.dim_orientation,
                    latent_dim=geometry_cfg.latent_dim,
                    xmin=jnp.array(
                        cfg.data.x_min if cfg.data.x_min is not None else [-1.0, -1.0],
                        dtype=jnp.float32,
                    ),
                    xmax=jnp.array(
                        cfg.data.x_max if cfg.data.x_max is not None else [1.0, 1.0],
                        dtype=jnp.float32,
                    ),
                    init_pos=False,
                    norm_angles=False,
                )
        else:
            raise ValueError("There is no Riemannian version in Jax implementation")

    elif geometry_cfg.input_space == "Spherical":

        if geometry_cfg.triv == "vanilla":
            if not meta:
                return VanillaOrthogonalLatents(
                    num_signals=num_signals,
                    num_latents=geometry_cfg.num_latents,
                    dim_signal=geometry_cfg.dim_signal,
                    dim_orientation=geometry_cfg.dim_orientation,
                    latent_dim=geometry_cfg.latent_dim,
                    norm_angles=False,
                )
            else:
                return VanillaMetaOrthogonalLatents(
                    num_signals=num_signals,
                    num_latents=geometry_cfg.num_latents,
                    dim_signal=geometry_cfg.dim_signal,
                    dim_orientation=geometry_cfg.dim_orientation,
                    latent_dim=geometry_cfg.latent_dim,
                    norm_angles=False,
                )
        else:
            raise ValueError("There is no Riemannian version in Jax implementation")

    else:
        raise ValueError()
