"""
Modifed from https://github.com/ArmanMaesumi/torchrbf
"""

import numpy as np
import warnings
import torch
import math

from itertools import combinations_with_replacement

from experiments.fitting.datasets.spherical.radial_fn import (
    SCALE_INVARIANT,
    RADIAL_FUNCS,
    MIN_DEGREE,
)


class RBFInterpolator(torch.nn.Module):
    """
    Radial basis function interpolator in Pytorch. This is a port of
    the RBFInterpolator from scipy.interpolate.RBFInterpolator. With
    GPU acceleration, this is much faster than the scipy version.
    SciPy reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html

    @param y: (n, d) tensor of data point coordinates
    @param d: (n, m) tensor of data vectors at y
    @param neighbors (optional): int [CURRENTLY UNIMPLEMENTED] specifies the
        number of neighbors to use for each interpolation point. If
        None, all points are used.
        Default is None.
    @param smoothing (optional): float or (n,) tensor of smoothing parameters
        Default is 0.0.
    @param kernel (optional): str, kernel function to use; one of
        ['linear', 'thin_plate_spline', 'cubic', 'quintic', 'gaussian'
        'multiquadric', 'inverse_multiquadric', 'inverse_quadratic']
        Default is 'thin_plate_spline'.
    @param epsilon (optional): float, shape parameter for the kernel function.
        If kernel is 'linear', 'thin_plate_spline', 'cubic', or
        'quintic', then default is 1.0 and can be ignored. Must be
        specified otherwise.
    @param degree (optional): int, degree of the polynomial added to the
        interpolation function. See scipy.interpolate.RBFInterpolator
        for more details.
    @param device (optional): str, specifies the default device to store tensors
        and perform interpolation.
    @param spherical (optional): bool, if True, assumes input points are on a unit sphere
        and uses geodesic distances. Points should be 3D cartesian coordinates that will
        be normalized to the unit sphere. Default is False.

    Returns a callable Torch Module that interpolates the data at given points.
    """

    def __init__(
        self,
        y,
        d,
        neighbors=None,
        smoothing=0.0,
        kernel="thin_plate_spline",
        epsilon=None,
        degree=None,
        device="cpu",
        spherical=False,
    ):
        super().__init__()

        if torch.backends.cuda.matmul.allow_tf32:
            warnings.warn(
                "TF32 is enabled, which may cause numerical issues in PyTorch RBFInterpolator. "
                "Consider disabling it with torch.backends.cuda.matmul.allow_tf32 = False",
                UserWarning,
            )

        self.device = device
        self.spherical = spherical

        # init:
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).to(device=device).float()

        if y.ndim != 2:
            raise ValueError("y must be a 2-dimensional tensor.")

        # For spherical mode, ensure 3D points and normalize
        if spherical:
            if y.shape[1] != 3:
                raise ValueError("For spherical mode, y must have shape (n, 3)")
            y = y / torch.norm(y, dim=1, keepdim=True)

        ny, ndim = y.shape
        if isinstance(d, np.ndarray):
            d = torch.from_numpy(d).to(device=device).float()

        if d.shape[0] != ny:
            raise ValueError(
                "The first dim of d must have the same length as the first dim of y."
            )

        d_shape = d.shape[1:]
        d = d.reshape((ny, -1))

        if isinstance(smoothing, (int, float)):
            smoothing = torch.full((ny,), smoothing, device=device).float()
        elif isinstance(smoothing, np.ndarray):
            smoothing = torch.Tensor(smoothing).to(device=device).float()
        elif not isinstance(smoothing, torch.Tensor):
            raise ValueError("`smoothing` must be a scalar or a 1-dimensional tensor.")

        kernel = kernel.lower()
        if kernel not in RADIAL_FUNCS:
            raise ValueError(f"Unknown kernel: {kernel}")

        if epsilon is None:
            if kernel in SCALE_INVARIANT:
                epsilon = 1.0
            else:
                raise ValueError("Must specify `epsilon` for this kernel.")
        else:
            epsilon = float(epsilon)

        min_degree = MIN_DEGREE.get(kernel, -1)
        if degree is None:
            degree = max(min_degree, 0)
        else:
            degree = int(degree)
            if degree < -1:
                raise ValueError("`degree` must be at least -1.")
            elif degree < min_degree:
                warnings.warn(
                    f"`degree` is too small for this kernel. Setting to {min_degree}.",
                    UserWarning,
                )

        if neighbors is None:
            nobs = ny
        else:
            raise ValueError("neighbors currently not supported")

        powers = monomial_powers(ndim, degree).to(device=device)
        if powers.shape[0] > nobs:
            raise ValueError("The data is not compatible with the requested degree.")

        if neighbors is None:
            shift, scale, coeffs = solve(
                y, d, smoothing, kernel, epsilon, powers, spherical
            )
            self.register_buffer("_shift", shift)
            self.register_buffer("_scale", scale)
            self.register_buffer("_coeffs", coeffs)

        self.register_buffer("y", y)
        self.register_buffer("d", d)
        self.register_buffer("smoothing", smoothing)
        self.register_buffer("powers", powers)

        self.d_shape = d_shape
        self.neighbors = neighbors
        self.kernel = kernel
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor):
        """
        Returns the interpolated data at the given points `x`.

        @param x: (n, d) tensor of points at which to query the interpolator
                  For spherical mode, should be (n, 3) cartesian coordinates
        @param use_grad (optional): bool, whether to use Torch autograd when
            querying the interpolator. Default is False.

        Returns a (n, m) tensor of interpolated data.
        """
        if x.ndim != 2:
            raise ValueError("`x` must be a 2-dimensional tensor.")

        nx, ndim = x.shape
        if ndim != self.y.shape[1]:
            raise ValueError(
                "Expected the second dim of `x` to have length " f"{self.y.shape[1]}."
            )

        # For spherical mode, normalize input points
        if self.spherical:
            x = x / torch.norm(x, dim=1, keepdim=True)

        kernel_func = RADIAL_FUNCS[self.kernel]

        yeps = self.y * self.epsilon
        xeps = x * self.epsilon

        # Only apply shift/scale in non-spherical mode
        if self.spherical:
            xhat = x
        else:
            xhat = (x - self._shift) / self._scale

        kv = kernel_vector(xeps, yeps, kernel_func, self.spherical)
        p = polynomial_matrix(xhat, self.powers)
        vec = torch.cat([kv, p], dim=1)
        out = torch.matmul(vec, self._coeffs)
        out = out.reshape((nx,) + self.d_shape)
        return out


def compute_distances(x, y, spherical=False):
    """
    Compute distances between points.

    @param x: (n, d) tensor
    @param y: (m, d) tensor
    @param spherical: if True, compute geodesic distances on unit sphere

    Returns: (n, m) distance matrix
    """
    if spherical:
        # Geodesic distance on unit sphere
        dots = torch.matmul(x, y.T)
        dots = torch.clamp(dots, -1.0, 1.0)  # Numerical safety
        return torch.arccos(dots)
    else:
        # Euclidean distance
        return torch.cdist(x, y)


def kernel_vector(x, y, kernel_func, spherical=False):
    """Evaluate radial functions with centers `y` for all points in `x`."""
    distances = compute_distances(x, y, spherical)
    return kernel_func(distances)


def polynomial_matrix(x, powers):
    """Evaluate monomials at `x` with given `powers`"""
    x_ = torch.repeat_interleave(x, repeats=powers.shape[0], dim=0)
    powers_ = powers.repeat(x.shape[0], 1)
    return torch.prod(x_**powers_, dim=1).view(x.shape[0], powers.shape[0])


def kernel_matrix(x, kernel_func, spherical=False):
    """Returns radial function values for all pairs of points in `x`."""
    distances = compute_distances(x, x, spherical)
    return kernel_func(distances)


def monomial_powers(ndim, degree):
    """Return the powers for each monomial in a polynomial.

    Parameters
    ----------
    ndim : int
        Number of variables in the polynomial.
    degree : int
        Degree of the polynomial.

    Returns
    -------
    (nmonos, ndim) int ndarray
        Array where each row contains the powers for each variable in a
        monomial.

    """
    nmonos = math.comb(degree + ndim, ndim)
    out = torch.zeros((nmonos, ndim), dtype=torch.int32)
    count = 0
    for deg in range(degree + 1):
        for mono in combinations_with_replacement(range(ndim), deg):
            for var in mono:
                out[count, var] += 1
            count += 1

    return out


def build(y, d, smoothing, kernel, epsilon, powers, spherical=False):
    """Build the RBF linear system"""

    p = d.shape[0]
    s = d.shape[1]
    r = powers.shape[0]
    kernel_func = RADIAL_FUNCS[kernel]

    # For spherical mode, skip shift/scale
    if spherical:
        shift = torch.zeros(y.shape[1], device=d.device)
        scale = torch.ones(y.shape[1], device=d.device)
        yhat = y
    else:
        mins = torch.min(y, dim=0).values
        maxs = torch.max(y, dim=0).values
        shift = (maxs + mins) / 2
        scale = (maxs - mins) / 2
        scale[scale == 0.0] = 1.0
        yhat = (y - shift) / scale

    yeps = y * epsilon

    lhs = torch.empty((p + r, p + r), device=d.device).float()
    lhs[:p, :p] = kernel_matrix(yeps, kernel_func, spherical)
    lhs[:p, p:] = polynomial_matrix(yhat, powers)
    lhs[p:, :p] = lhs[:p, p:].T
    lhs[p:, p:] = 0.0
    lhs[:p, :p] += torch.diag(smoothing)

    rhs = torch.empty((r + p, s), device=d.device).float()
    rhs[:p] = d
    rhs[p:] = 0.0

    return lhs, rhs, shift, scale


def solve(y, d, smoothing, kernel, epsilon, powers, spherical=False):
    """Build then solve the RBF linear system"""

    lhs, rhs, shift, scale = build(y, d, smoothing, kernel, epsilon, powers, spherical)
    try:
        coeffs = torch.linalg.solve(lhs, rhs)
    except RuntimeError:  # singular matrix
        msg = "Singular matrix."
        nmonos = powers.shape[0]
        if nmonos > 0:
            if spherical:
                pmat = polynomial_matrix(y, powers)
            else:
                pmat = polynomial_matrix((y - shift) / scale, powers)
            rank = torch.linalg.matrix_rank(pmat)
            if rank < nmonos:
                msg = (
                    "Singular matrix. The matrix of monomials evaluated at "
                    "the data point coordinates does not have full column "
                    f"rank ({rank}/{nmonos})."
                )

        raise ValueError(msg)

    return shift, scale, coeffs
