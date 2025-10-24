from equiv_eikonal.steerable_attention.threeway_invariants.position_orientation_input.position_orientation_latent import *
from equiv_eikonal.steerable_attention.threeway_invariants.position_orientation_input.group_latent import *


def get_position_orientation_invariants(
    geometry_cfg,
):
    """Get the invariant for the attention module.

    Args:
        name (str): The name of the invariant.

    Returns:
        BaseInvariant: The invariant module.

    """

    if "SE" in geometry_cfg.group:
        if "Perm" in geometry_cfg.group:
            if geometry_cfg.dim_orientation == 2:
                return SymSpecialEuclideanR2xS1InputsSE2Latent()
            else:
                return SymSpecialEuclideanR2xS1InputsR2xS1Latent()

        else:
            if geometry_cfg.dim_orientation == 2:
                return SpecialEuclideanR2xS1InputsSE2Latent()
            else:
                return SpecialEuclideanR2xS1InputsR2xS1Latent()

    else:
        if "Perm" in geometry_cfg.group:
            if geometry_cfg.dim_orientation == 2:
                raise ValueError(
                    "E(n) has two connected components, making it unfeasable to optimize the latents in these cases."
                )
            else:
                return SymEuclideanR2xS1InputsR2xS1Latent()
        else:
            if geometry_cfg.dim_orientation == 2:
                raise ValueError(
                    "E(n) has two connected components, making it unfeasable to optimize the latents in these cases."
                )
            else:
                return EuclideanR2xS1InputsR2xS1Latent()
