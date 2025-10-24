from equiv_eikonal.steerable_attention.threeway_invariants._base_invariant import (
    BaseThreewayInvariants,
)
from equiv_eikonal.steerable_attention.threeway_invariants.euclidean_input.euclidean_latent import *
from equiv_eikonal.steerable_attention.threeway_invariants.euclidean_input.position_orientation_latent import *
from equiv_eikonal.steerable_attention.threeway_invariants.euclidean_input.group_latent import *
from equiv_eikonal.steerable_attention.threeway_invariants.euclidean_input.no_equiv import *


def get_euclidean_invariants(geometry_cfg) -> BaseThreewayInvariants:
    """Get the invariant for the attention module.

    Args:
        name (str): The name of the invariant.

    Returns:
        BaseInvariant: The invariant module.

    """
    if "NoEquiv" in geometry_cfg.group:
        if "Perm" in geometry_cfg.group:
            if geometry_cfg.dim_signal == 2:
                return SymNoEquivR2Inputs()
            else:
                return SymNoEquivR3Inputs()
        else:
            if geometry_cfg.dim_signal == 2:
                return NoEquivR2Inputs()
            else:
                return NoEquivR3Inputs()

    elif "SE" in geometry_cfg.group:
        if "Perm" in geometry_cfg.group:
            if geometry_cfg.dim_signal == 2:
                if geometry_cfg.dim_orientation == 2:
                    return SymSpecialEuclideanR2InputsSE2Latent()
                elif geometry_cfg.dim_orientation == 1:
                    return SymSpecialEuclideanR2InputsR2xS1Latent()
                else:
                    return SymSpecialEuclideanR2InputsR2Latent()
            else:
                if geometry_cfg.dim_orientation == 3:
                    return SymSpecialEuclideanR3InputsSE3Latent()
                elif geometry_cfg.dim_orientation == 2:
                    raise ValueError(
                        "Stiefel latents are not available in JAX implementation"
                    )
                elif geometry_cfg.dim_orientation == 1:
                    return SymSpecialEuclideanR3InputsR3xS2Latent()
                else:
                    return SymSpecialEuclideanR3InputsR3Latent()
        else:
            if geometry_cfg.dim_signal == 2:
                if geometry_cfg.dim_orientation == 2:
                    return SpecialEuclideanR2InputsSE2Latent()
                elif geometry_cfg.dim_orientation == 1:
                    return SpecialEuclideanR2InputsR2xS1Latent()
                else:
                    return SpecialEuclideanR2InputsR2Latent()
            else:
                if geometry_cfg.dim_orientation == 3:
                    return SpecialEuclideanR3InputsSE3Latent()
                elif geometry_cfg.dim_orientation == 2:
                    raise ValueError(
                        "Stiefel latents are not available in JAX implementation"
                    )
                elif geometry_cfg.dim_orientation == 1:
                    return SpecialEuclideanR3InputsR3xS2Latent()
                else:
                    return SpecialEuclideanR3InputsR3Latent()
    else:
        if "Perm" in geometry_cfg.group:
            if geometry_cfg.dim_signal == 2:
                if geometry_cfg.dim_orientation == 2:
                    raise ValueError(
                        "E(n) has two connected components, making it unfeasable to optimize the latents in these cases."
                    )
                elif geometry_cfg.dim_orientation == 1:
                    return SymEuclideanR2InputsR2xS1Latent()
                else:
                    return SymEuclideanR2InputsR2Latent()
            else:
                if geometry_cfg.dim_orientation == 3:
                    raise ValueError(
                        "E(n) has two connected components, making it unfeasable to optimize the latents in these cases."
                    )
                elif geometry_cfg.dim_orientation == 2:
                    raise ValueError(
                        "Stiefel latents are not available in JAX implementation"
                    )
                elif geometry_cfg.dim_orientation == 1:
                    return SymEuclideanR3InputsR3xS2Latent()
                else:
                    return SymEuclideanR3InputsR3Latent()
        else:
            if geometry_cfg.dim_signal == 2:
                if geometry_cfg.dim_orientation == 2:
                    raise ValueError(
                        "E(n) has two connected components, making it unfeasable to optimize the latents in these cases."
                    )
                elif geometry_cfg.dim_orientation == 1:
                    return EuclideanR2InputsR2xS1Latent()
                else:
                    return EuclideanR2InputsR2Latent()
            else:
                if geometry_cfg.dim_orientation == 3:
                    raise ValueError(
                        "E(n) has two connected components, making it unfeasable to optimize the latents in these cases."
                    )
                elif geometry_cfg.dim_orientation == 2:
                    raise ValueError(
                        "Stiefel latents are not available in JAX implementation"
                    )
                elif geometry_cfg.dim_orientation == 1:
                    return EuclideanR3InputsR3xS2Latent()
                else:
                    return EuclideanR3InputsR3Latent()
