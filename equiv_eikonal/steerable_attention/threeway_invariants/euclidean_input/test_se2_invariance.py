"""
Invariance test for SpecialEuclideanR2InputsSE2Latent class.

This test verifies that the class produces invariant outputs when SE(2) group actions
are applied to both inputs and latents.

SE(2) group action consists of:
- Rotation by angle theta (via rotation matrix R)
- Translation by vector t

For inputs: (x, y) -> (R @ x + t, R @ y + t)
For latents: (p_pos, p_ori) -> (R @ p_pos + t, R @ p_ori @ R^T)
"""

import jax
import jax.numpy as jnp
import numpy as np
from equiv_eikonal.steerable_attention.threeway_invariants.euclidean_input.group_latent import (
    SpecialEuclideanR2InputsSE2Latent,
)


def create_se2_rotation_matrix(theta):
    """Create a 2D rotation matrix for angle theta.

    Args:
        theta: Rotation angle in radiansa

    Returns:
        2x2 rotation matrix
    """
    return jnp.array(
        [[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]]
    )


def apply_se2_to_inputs(inputs, R, t):
    """Apply SE(2) transformation to input points.

    Args:
        inputs: Shape (batch_size, num_sample_pairs, 2, 2)
                First dimension index 0 is x point, index 1 is y point
        R: Rotation matrix (2, 2)
        t: Translation vector (2,)

    Returns:
        Transformed inputs with same shape
    """
    # inputs[:, :, 0, :] are x points, inputs[:, :, 1, :] are y points
    # Apply rotation and translation: R @ point + t
    x_transformed = jnp.einsum("ij,bsj->bsi", R, inputs[:, :, 0, :]) + t
    y_transformed = jnp.einsum("ij,bsj->bsi", R, inputs[:, :, 1, :]) + t

    # Stack back to original format
    transformed_inputs = jnp.stack([x_transformed, y_transformed], axis=2)
    return transformed_inputs


def apply_se2_to_latents(p, R, t):
    """Apply SE(2) transformation to latent poses.

    For SE(2) group:
    - Position transforms as: p_pos -> R @ p_pos + t
    - Orientation transforms as: p_ori -> R @ p_ori (composition of rotations)

    Note: For SE(2) (special Euclidean, rotations only), we use composition R @ p_ori,
    not conjugation R @ p_ori @ R^T (which would be for E(2) with reflections).

    Args:
        p: Tuple of (p_pos, p_ori)
           p_pos: Shape (batch_size, num_latents, 2)
           p_ori: Shape (batch_size, num_latents, 2, 2)
        R: Rotation matrix (2, 2)
        t: Translation vector (2,)

    Returns:
        Transformed latent poses (p_pos_new, p_ori_new)
    """
    p_pos, p_ori = p

    # Transform positions: R @ p_pos + t
    p_pos_transformed = jnp.einsum("ij,bnj->bni", R, p_pos) + t

    # Transform orientations: R @ p_ori (composition of rotations)
    # p_ori shape is (B, N, 2, 2)
    # We want: R @ p_ori[b,n] for each b, n
    p_ori_transformed = jnp.einsum("ij,bnjk->bnik", R, p_ori)

    return (p_pos_transformed, p_ori_transformed)


def test_se2_invariance(
    batch_size=1,
    num_sample_pairs=1,
    num_latents=1,
    theta_degrees=45,
    translation_x=0.5,
    translation_y=1.0,
    seed=42,
    verbose=True,
):
    """Test SE(2) invariance of SpecialEuclideanR2InputsSE2Latent.

    Args:
        batch_size: Batch size
        num_sample_pairs: Number of sample pairs
        num_latents: Number of latents
        theta_degrees: Rotation angle in degrees
        translation_x: Translation in x direction
        translation_y: Translation in y direction
        seed: Random seed
        verbose: Whether to print detailed output

    Returns:
        bool: True if invariance test passes
    """
    key = jax.random.PRNGKey(seed)

    # Create arbitrary inputs and latents
    B, S, N = batch_size, num_sample_pairs, num_latents

    # Inputs shape: (B, S, 2, 2)
    # First 2: [x_point, y_point], Second 2: coordinates in R^2
    inputs = jax.random.normal(key, (B, S, 2, 2))

    # Latent positions: (B, N, 2)
    p_pos = jax.random.normal(jax.random.split(key)[0], (B, N, 2))

    # Latent orientations: (B, N, 2, 2) - should be rotation matrices
    # Generate random rotation matrices by creating random angles
    angles = jax.random.uniform(
        jax.random.split(key)[1], (B, N), minval=0, maxval=2 * np.pi
    )
    p_ori = jnp.stack(
        [
            jnp.stack([jnp.cos(angles), -jnp.sin(angles)], axis=-1),
            jnp.stack([jnp.sin(angles), jnp.cos(angles)], axis=-1),
        ],
        axis=-2,
    )

    p = (p_pos, p_ori)

    # Create SE(2) transformation
    theta = jnp.deg2rad(theta_degrees)
    R = create_se2_rotation_matrix(theta)
    t = jnp.array([translation_x, translation_y])

    # Apply SE(2) to inputs and latents
    inputs_transformed = apply_se2_to_inputs(inputs, R, t)
    p_transformed = apply_se2_to_latents(p, R, t)

    # Initialize the invariant function
    invariant_fn = SpecialEuclideanR2InputsSE2Latent()

    # Compute invariants for original and transformed inputs
    output_original, params = invariant_fn.init_with_output(key, inputs, p)
    output_transformed = invariant_fn.apply(params, inputs_transformed, p_transformed)

    # Check if outputs are equal (invariant)
    max_diff = jnp.max(jnp.abs(output_original - output_transformed))
    relative_error = max_diff / (jnp.max(jnp.abs(output_original)) + 1e-10)

    if verbose:
        print("=" * 70)
        print("SE(2) Invariance Test for SpecialEuclideanR2InputsSE2Latent")
        print("=" * 70)
        print(f"\nTest Parameters:")
        print(f"  Batch size: {B}")
        print(f"  Num sample pairs: {S}")
        print(f"  Num latents: {N}")
        print(f"  Rotation angle: {theta_degrees}°")
        print(f"  Translation: ({translation_x}, {translation_y})")
        print(f"\nInput shapes:")
        print(f"  inputs: {inputs.shape}")
        print(f"  p_pos: {p_pos.shape}")
        print(f"  p_ori: {p_ori.shape}")
        print(f"\nOutput shape: {output_original.shape}")
        print(f"\nInvariance check:")
        print(f"  Max absolute difference: {max_diff:.2e}")
        print(f"  Relative error: {relative_error:.2e}")
        print(f"  Max output value: {jnp.max(jnp.abs(output_original)):.2e}")

        # Print some sample values
        print(
            f"\nSample original output (first element): {output_original[0, 0, 0, :]}"
        )
        print(
            f"Sample transformed output (first element): {output_transformed[0, 0, 0, :]}"
        )

        threshold = 1e-5
        if max_diff < threshold:
            print(f"\n✓ PASS: Outputs are invariant (max_diff < {threshold})")
        else:
            print(f"\n✗ FAIL: Outputs are NOT invariant (max_diff >= {threshold})")
        print("=" * 70)

    # Test passes if max difference is very small
    return max_diff < 1e-5


def test_multiple_transformations(verbose=True):
    """Test invariance under multiple different SE(2) transformations."""
    test_cases = [
        {
            "theta_degrees": 0,
            "translation_x": 0.0,
            "translation_y": 0.0,
            "name": "Identity",
        },
        {
            "theta_degrees": 90,
            "translation_x": 0.0,
            "translation_y": 0.0,
            "name": "90° rotation",
        },
        {
            "theta_degrees": 180,
            "translation_x": 0.0,
            "translation_y": 0.0,
            "name": "180° rotation",
        },
        {
            "theta_degrees": 0,
            "translation_x": 1.0,
            "translation_y": 2.0,
            "name": "Translation only",
        },
        {
            "theta_degrees": 45,
            "translation_x": 0.5,
            "translation_y": 1.0,
            "name": "45° rot + trans",
        },
        {
            "theta_degrees": -30,
            "translation_x": -1.5,
            "translation_y": 0.7,
            "name": "-30° rot + trans",
        },
    ]

    all_passed = True
    results = []

    for i, test_case in enumerate(test_cases):
        if verbose:
            print(f"\n\nTest {i+1}/{len(test_cases)}: {test_case['name']}")
            print("-" * 70)

        passed = test_se2_invariance(
            batch_size=2,
            num_sample_pairs=3,
            num_latents=2,
            theta_degrees=test_case["theta_degrees"],
            translation_x=test_case["translation_x"],
            translation_y=test_case["translation_y"],
            verbose=verbose,
            seed=42 + i,
        )

        results.append((test_case["name"], passed))
        all_passed = all_passed and passed

    if verbose:
        print("\n\n" + "=" * 70)
        print("SUMMARY OF ALL TESTS")
        print("=" * 70)
        for name, passed in results:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status}: {name}")
        print("=" * 70)
        if all_passed:
            print("\n🎉 ALL TESTS PASSED!")
        else:
            print("\n❌ SOME TESTS FAILED")
        print("=" * 70)

    return all_passed


if __name__ == "__main__":
    # Run comprehensive test suite
    all_passed = test_multiple_transformations(verbose=True)

    # Exit with appropriate code
    import sys

    sys.exit(0 if all_passed else 1)
