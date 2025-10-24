from equiv_eikonal.steerable_attention.threeway_invariants._base_invariant import (
    BaseThreewayInvariants,
)
from equiv_eikonal.steerable_attention.threeway_invariants.euclidean_input import (
    get_euclidean_invariants,
)
from equiv_eikonal.steerable_attention.threeway_invariants.position_orientation_input import (
    get_position_orientation_invariants,
)

from equiv_eikonal.steerable_attention.threeway_invariants.spherical_input import (
    get_spherical_invariants,
)


def get_invariants(cfg) -> BaseThreewayInvariants:
    """Get the invariant for the attention module.

    Args:
        name (str): The name of the invariant.

    Returns:
        BaseInvariant: The invariant module.

    """
    if cfg.input_space == "Euclidean":
        return get_euclidean_invariants(cfg)
    elif cfg.input_space == "Position_Orientation":
        return get_position_orientation_invariants(cfg)
    elif cfg.input_space == "Spherical":
        return get_spherical_invariants(cfg)
    else:
        raise ValueError()
