from equiv_eikonal.steerable_attention.threeway_invariants._base_invariant import (
    BaseThreewayInvariants,
)
from equiv_eikonal.steerable_attention.threeway_invariants.spherical_input.spherical_latent import *
from equiv_eikonal.steerable_attention.threeway_invariants.spherical_input.group_latent import *


def get_spherical_invariants(geometry_cfg) -> BaseThreewayInvariants:
    """Get the invariant for the attention module.

    Args:
        name (str): The name of the invariant.

    Returns:
        BaseInvariant: The invariant module.

    """
    if "NoEquiv" in geometry_cfg.group:
        if "Perm" in geometry_cfg.group:
            if geometry_cfg.dim_signal == 2:
                if geometry_cfg.dim_orientation == 1:
                    return SymNoEquivS1InputsS1Latent()
                else:
                    raise ValueError()
            else:
                if geometry_cfg.dim_orientation == 1:
                    return SymNoEquivS2InputsS2Latent()
                else:
                    raise ValueError()
        else:
            if geometry_cfg.dim_signal == 2:
                if geometry_cfg.dim_orientation == 1:
                    return NoEquivS1InputsS1Latent()
                else:
                    raise ValueError()
            else:
                if geometry_cfg.dim_orientation == 1:
                    return NoEquivS2InputsS2Latent()
                else:
                    raise ValueError()

    elif "SO" in geometry_cfg.group:
        if "Perm" in geometry_cfg.group:
            if geometry_cfg.dim_signal == 2:
                if geometry_cfg.dim_orientation == 2:
                    return SymSpecialOrthogonalS1InputsSO2Latent()
                elif geometry_cfg.dim_orientation == 1:
                    return SymSpecialOrthogonalSnInputsSnLatent()
            else:
                if geometry_cfg.dim_orientation == 3:
                    return SymSpecialOrthogonalS2InputsSO3Latent()
                elif geometry_cfg.dim_orientation == 2:
                    # this is a special case in which we have only rotations over the z-axis
                    # we embed the z-axis rotation in SO(3) as a 2D rotation
                    return SymSpecialOrthogonalS2InputsSO3Latent()
                else:
                    return SymSpecialOrthogonalSnInputsSnLatent()
        else:
            if geometry_cfg.dim_signal == 2:
                if geometry_cfg.dim_orientation == 2:
                    return SpecialOrthogonalS1InputsSO2Latent()
                elif geometry_cfg.dim_orientation == 1:
                    return SpecialOrthogonalSnInputsSnLatent()
            else:
                if geometry_cfg.dim_orientation == 3:
                    return SpecialOrthogonalS2InputsSO3Latent()
                elif geometry_cfg.dim_orientation == 2:
                    # this is a special case in which we have only rotations over the z-axis
                    # we embed the z-axis rotation in SO(3) as a 2D rotation
                    return SpecialOrthogonalS2InputsSO3Latent()
                else:
                    return SpecialOrthogonalSnInputsSnLatent()
    elif "O" in geometry_cfg.group:
        if "Perm" in geometry_cfg.group:
            if geometry_cfg.dim_signal == 2:
                if geometry_cfg.dim_orientation == 1:
                    return SymOrthogonalSnInputsSnLatent()
                else:
                    raise ValueError(
                        "Unsupported configuration for spherical inputs. O(n) has two connected components, making it unfeasable to optimize the latents in these cases."
                    )
            else:
                if geometry_cfg.dim_orientation == 1:
                    return SymOrthogonalSnInputsSnLatent()
                else:
                    raise ValueError(
                        "Unsupported configuration for spherical inputs. O(n) has two connected components, making it unfeasable to optimize the latents in these cases."
                    )
        else:
            if geometry_cfg.dim_signal == 2:
                if geometry_cfg.dim_orientation == 1:
                    return OrthogonalSnInputsSnLatent()
                else:
                    raise ValueError(
                        "Unsupported configuration for spherical inputs. O(n) has two connected components, making it unfeasable to optimize the latents in these cases."
                    )
            else:
                if geometry_cfg.dim_orientation == 1:
                    return OrthogonalSnInputsSnLatent()
                else:
                    raise ValueError(
                        "Unsupported configuration for spherical inputs. O(n) has two connected components, making it unfeasable to optimize the latents in these cases."
                    )
    else:
        raise ValueError(
            f"Unsupported group {geometry_cfg.group} for spherical inputs."
        )
