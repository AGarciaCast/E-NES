import numpy as np
from eikonalfm import factored_fast_marching as euclidean_ffm
from eikonalfm import distance as euclidean_dist
import time


import math
import cupy as cp


import taichi as ti

from experiments.fitting.utils.ground_truth.IterativeEikonal.eikivp.SE2.Riemannian.distancemap import (
    eikonal_solver as eikonal_solver_se_2_riem,
)

from experiments.fitting.utils.ground_truth.IterativeEikonal.eikivp.SE2.subRiemannian.distancemap import (
    eikonal_solver as eikonal_solver_se_2_sub,
)


import math


def create_grid_data_euclidean_2D(
    nx, ny, x_min=None, x_max=None, skip_r=1, skip_s=5, top_benchmark=False
):

    xmin, ymin = x_min if x_min is not None else [-1.0, -1.0]
    xmax, ymax = x_max if x_max is not None else [1.0, 1.0]
    x = np.linspace(xmin, xmax, nx, dtype=np.float32)
    y = np.linspace(ymin, ymax, ny, dtype=np.float32)

    xr = x[::skip_r]
    yr = y[::skip_r]

    Xr = np.stack(np.meshgrid(xr, yr, indexing="ij"), axis=-1)

    if top_benchmark:

        selected_pts_x = range(0, nx, nx // 5)
        xs = x[selected_pts_x]

        ys = np.repeat(y[-1], 1, axis=0)
        selected_pts_y = np.repeat(len(y) - 1, 1, axis=0)
    else:
        selected_pts_x = range(0, nx, skip_s)
        selected_pts_y = range(0, ny, skip_s)
        xs = x[::skip_s]
        ys = y[::skip_s]

    Xs = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1)

    # coords = pair_tensors(torch.from_numpy(Xs).view(-1, 2).unsqueeze(0),
    #                                  torch.from_numpy(Xr).view(-1, 2).unsqueeze(0))

    coords = np.stack(np.meshgrid(xs, ys, xr, yr, indexing="ij"), axis=-1).reshape(
        -1, 2, 2
    )
    coords = coords[None, ...]

    grid_data = {
        "nx": nx,
        "ny": ny,
        "x": x,
        "y": y,
        "skip_r": skip_r,
        "xr": xr,
        "yr": yr,
        "skip_s": skip_s,
        "xs": xs,
        "ys": ys,
        "Xr": Xr,
        "Xs": Xs,
        "selected_pts_x": selected_pts_x,
        "selected_pts_y": selected_pts_y,
        "coords": coords,
    }
    return grid_data


def euclidean_2D_ffm(Vel, grid_data):
    if not isinstance(Vel, np.ndarray):
        Vel = Vel.detach().cpu().numpy()

    assert np.all(Vel >= 0.0)

    xs = grid_data["xs"]
    ys = grid_data["ys"]
    xr = grid_data["xr"]
    yr = grid_data["xr"]
    selected_pts_x = grid_data["selected_pts_x"]
    selected_pts_y = grid_data["selected_pts_y"]

    # Traveltime using Factored fast marching of second order
    T_ref = np.empty((len(xs), len(ys), len(xr), len(yr)))
    dxs = [xr[1] - xr[0], yr[1] - yr[0]]
    start_time = time.time()

    for i, ixs in enumerate(selected_pts_x):
        for j, jys in enumerate(selected_pts_y):
            T_ref[i, j] = euclidean_ffm(Vel, (ixs, jys), dxs, 2)
            T_ref[i, j] *= euclidean_dist(Vel.shape, dxs, (ixs, jys), indexing="ij")

    fmmTime = time.time() - start_time

    return T_ref, fmmTime, grid_data


def create_grid_data_euclidean_3D(
    nx, ny, nz, x_min=None, x_max=None, skip_r=1, skip_s=5
):

    xmin, ymin, zmin = x_min if x_min is not None else [-1.0, -1.0, -1.0]
    xmax, ymax, zmax = x_max if x_max is not None else [1.0, 1.0, 1.0]
    x = np.linspace(xmin, xmax, nx, dtype=np.float32)
    y = np.linspace(ymin, ymax, ny, dtype=np.float32)
    z = np.linspace(zmin, zmax, nz, dtype=np.float32)

    xr = x[::skip_r]
    yr = y[::skip_r]
    zr = z[::skip_r]

    Xr = np.stack(np.meshgrid(xr, yr, zr, indexing="ij"), axis=-1)

    xs = x[::skip_s]
    ys = y[::skip_s]
    zs = z[::skip_s]

    Xs = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)

    coords = np.stack(
        np.meshgrid(xs, ys, zs, xr, yr, zr, indexing="ij"), axis=-1
    ).reshape(-1, 2, 3)
    coords = coords[None, ...]

    grid_data = {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "x": x,
        "y": y,
        "z": z,
        "skip_r": skip_r,
        "xr": xr,
        "yr": yr,
        "zr": zr,
        "skip_s": skip_s,
        "xs": xs,
        "ys": ys,
        "zs": zs,
        "Xr": Xr,
        "Xs": Xs,
        "coords": coords,
    }

    return grid_data


def euclidean_3D_ffm(Vel, grid_data):
    if not isinstance(Vel, np.ndarray):
        Vel = Vel.detach().cpu().numpy()

    nx = grid_data["nx"]
    ny = grid_data["ny"]
    nz = grid_data["nz"]
    xs = grid_data["xs"]
    ys = grid_data["ys"]
    zs = grid_data["zs"]
    xr = grid_data["xr"]
    yr = grid_data["yr"]
    zr = grid_data["zr"]
    skip_s = grid_data["skip_s"]

    # Traveltime using Factored fast marching of second order
    T_ref = np.empty((len(xs), len(ys), len(zs), len(xr), len(yr), len(zr)))
    dxs = [xr[1] - xr[0], yr[1] - yr[0], zr[1] - zr[0]]
    start_time = time.time()

    for i, ixs in enumerate(range(0, nx, skip_s)):
        for j, jys in enumerate(range(0, ny, skip_s)):
            for k, kzs in enumerate(range(0, nz, skip_s)):
                T_ref[i, j, k] = euclidean_ffm(Vel, (ixs, jys, kzs), dxs, 2)
                T_ref[i, j, k] *= euclidean_dist(
                    Vel.shape, dxs, (ixs, jys, kzs), indexing="ij"
                )

    fmmTime = time.time() - start_time

    return T_ref, fmmTime, grid_data


def create_grid_data_position_orientation(
    nx,
    ny,
    ntheta,
    x_min=None,
    x_max=None,
    skip_r=1,
    skip_s=5,
    skip_s_theta=5,
    theta_range="zero",
):

    theta_range = (
        [
            0.0,
            2.0 * math.pi,
        ]
        if theta_range == "zero"
        else [-math.pi, math.pi]
    )

    xmin, ymin, thetamin = (
        x_min + [theta_range[0]]
        if x_min is not None
        else [-1.0, -1.0] + [theta_range[0]]
    )
    xmax, ymax, thetamax = (
        x_max + [theta_range[1]] if x_max is not None else [1.0, 1.0] + [theta_range[1]]
    )
    x = np.linspace(xmin, xmax, nx, dtype=np.float32)
    y = np.linspace(ymin, ymax, ny, dtype=np.float32)
    theta = np.linspace(thetamin, thetamax, ntheta, endpoint=False, dtype=np.float32)
    xr = x[::skip_r]
    yr = y[::skip_r]
    thetar = theta[::skip_r]

    Xr = np.stack(np.meshgrid(xr, yr, thetar, indexing="ij"), axis=-1)

    xs = x[::skip_s]
    ys = y[::skip_s]
    thetas = theta[::skip_s_theta]

    Xs = np.stack(np.meshgrid(xs, ys, thetas, indexing="ij"), axis=-1)

    coords = np.stack(
        np.meshgrid(xs, ys, thetas, xr, yr, thetar, indexing="ij"), axis=-1
    ).reshape(-1, 2, 3)
    coords = coords[None, ...]

    grid_data = {
        "nx": nx,
        "ny": ny,
        "ntheta": ntheta,
        "x": x,
        "y": y,
        "theta": theta,
        "skip_r": skip_r,
        "xr": xr,
        "yr": yr,
        "thetar": thetar,
        "skip_s": skip_s,
        "skip_s_theta": skip_s_theta,
        "xs": xs,
        "ys": ys,
        "thetas": thetas,
        "Xr": Xr,
        "Xs": Xs,
        "coords": coords,
    }

    return grid_data


def position_orientation_taichi(
    Vel,
    grid_data,
    device="cpu",
    xi=1.0,
    epsilon=1.0,
    sub_riem=False,
    n_max=1e5,
    n_max_initialisation=1e4,
    n_check=2e3,
    n_check_initialisation=2e3,
    tol=1e-3,
    initial_condition=200.0,
):

    if not isinstance(Vel, np.ndarray):
        Vel = Vel.detach().cpu().numpy()

    nx = grid_data["nx"]
    ny = grid_data["ny"]
    ntheta = grid_data["ntheta"]
    xs = grid_data["xs"]
    ys = grid_data["ys"]
    thetas = grid_data["thetas"]
    xr = grid_data["xr"]
    yr = grid_data["yr"]
    thetar = grid_data["thetar"]
    Xr = grid_data["Xr"]
    skip_s = grid_data["skip_s"]
    skip_s_theta = grid_data["skip_s_theta"]

    Cost = 1.0 / Vel  # Cost is from (0, 1]
    dtheta = thetar[1] - thetar[0]
    dxy = xr[1] - xr[0]
    thetarGrid = Xr[..., 2]

    T_ref = np.empty((len(xs), len(ys), len(thetas), len(xr), len(yr), len(thetar)))

    sub_riem = sub_riem or epsilon == 0.0
    if not sub_riem:
        G_np_SE2 = np.array((xi**2, xi**2 / epsilon**2, 1.0), dtype=np.float32)

    fmmTime = 0

    for i, ixs in enumerate(range(0, nx, skip_s)):
        for j, jys in enumerate(range(0, ny, skip_s)):
            for k, kzs in enumerate(range(0, ntheta, skip_s_theta)):
                ti.init(
                    arch=ti.cpu if device == "cpu" else ti.gpu,
                    debug=False,
                    log_level=ti.ERROR,
                )

                start_time = time.time()
                if sub_riem:
                    W = eikonal_solver_se_2_sub(
                        Cost,
                        (np.int32(ixs), np.int32(jys), np.int32(kzs)),
                        xi,
                        dxy,
                        dtheta,
                        thetarGrid,
                        n_max=n_max,
                        n_max_initialisation=n_max_initialisation,
                        n_check=n_check,
                        n_check_initialisation=n_check_initialisation,
                        tol=tol,
                        initial_condition=initial_condition,
                    )

                else:
                    W = eikonal_solver_se_2_riem(
                        Cost,
                        (np.int32(ixs), np.int32(jys), np.int32(kzs)),
                        G_np_SE2,
                        dxy,
                        dtheta,
                        thetarGrid,
                        n_max=n_max,
                        n_max_initialisation=n_max_initialisation,
                        n_check=n_check,
                        n_check_initialisation=n_check_initialisation,
                        tol=tol,
                        initial_condition=initial_condition,
                    )

                T_ref[i, j, k] = W

                fmmTime += time.time() - start_time

                ti.reset()

    return T_ref, fmmTime, grid_data


from agd import Eikonal
from agd.Metrics import Riemann

Eikonal.dictIn.default_mode = "gpu"


def create_grid_data_spherical(
    nx,
    ny,
    x_min=None,
    x_max=None,
    skip_r=1,
    skip_s=5,
):
    assert skip_r == 1, "skip_r must be 1 for spherical coordinates"

    xmin, ymin = x_min if x_min is not None else [0, 0 * np.pi / 180]
    xmax, ymax = x_max if x_max is not None else [2 * np.pi, np.pi - 0 * np.pi / 180]

    hfmIn = Eikonal.dictIn(
        {
            "model": "Riemann2",  # Two-dimensional Riemannian eikonal equation
            "periodic": (True, False),
        }
    )

    hfmIn.SetRect(sides=[[xmin, xmax], [ymin, ymax]], dims=[nx, ny])

    X, Y = hfmIn.Grid()
    X = cp.asnumpy(X)
    Y = cp.asnumpy(Y)

    x = X[:, 0]
    y = Y[0, :]

    xr = x[::skip_r]
    yr = y[::skip_r]

    Xr = np.stack(np.meshgrid(xr, yr, indexing="ij"), axis=-1)

    selected_pts_x = range(0, nx, skip_s)
    selected_pts_y = range(0, ny, skip_s)
    xs = x[::skip_s]
    ys = y[::skip_s]

    Xs = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1)

    # coords = pair_tensors(torch.from_numpy(Xs).view(-1, 2).unsqueeze(0),
    #                                  torch.from_numpy(Xr).view(-1, 2).unsqueeze(0))

    coords = np.stack(np.meshgrid(xs, ys, xr, yr, indexing="ij"), axis=-1).reshape(
        -1, 2, 2
    )
    coords = coords[None, ...]

    grid_data = {
        "nx": nx,
        "ny": ny,
        "x": x,
        "y": y,
        "skip_r": skip_r,
        "xr": xr,
        "yr": yr,
        "skip_s": skip_s,
        "xs": xs,
        "ys": ys,
        "Xr": Xr,
        "Xs": Xs,
        "selected_pts_x": selected_pts_x,
        "selected_pts_y": selected_pts_y,
        "coords": coords,
        "sides": [[xmin, xmax], [ymin, ymax]],
        "dims": [nx, ny],
    }
    return grid_data


def spherical_agd(Vel, grid_data):
    if not isinstance(Vel, np.ndarray):
        Vel = Vel.detach().cpu().numpy()

    assert np.all(Vel >= 0.0)

    xs = grid_data["xs"]
    ys = grid_data["ys"]
    Xs = grid_data["Xs"]
    xr = grid_data["xr"]
    yr = grid_data["xr"]

    sides = grid_data["sides"]
    dims = grid_data["dims"]

    hfmIn = Eikonal.dictIn(
        {
            "model": "Riemann2",  # Two-dimensional Riemannian eikonal equation
            "seedValue": 0,  # Can be omitted, since this is the default.
            "periodic": (True, False),
        }
    )

    hfmIn.SetRect(sides=sides, dims=dims)
    hfmIn["order"] = 2
    hfmIn["speed"] = Vel
    X, Y = hfmIn.Grid()
    hfmIn["metric"] = Riemann(
        [[np.sin(Y) ** 2, np.zeros_like(X)], [np.zeros_like(X), np.ones_like(X)]]
    )
    hfmIn["exportValues"] = 1

    # Traveltime using Factored fast marching of second order
    T_ref = np.empty((len(xs), len(ys), len(xr), len(yr)))
    start_time = time.time()

    for i in range(len(xs)):
        for j in range(len(ys)):
            hfmIn["seed"] = Xs[i, j]
            T_ref[i, j] = cp.asnumpy(hfmIn.Run()["values"].get())

    fmmTime = time.time() - start_time

    return T_ref, fmmTime, grid_data
