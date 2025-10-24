from functools import partial
from experiments.fitting.utils.visualization.visualization_rec import (
    visualize_reconstructions_euclidean_2D,
    visualize_reconstructions_euclidean_3D,
    visualize_reconstructions_spherical,
)

from experiments.fitting.utils.visualization.visualization_gt import (
    visualize_gt_euclidean_2D,
    visualize_gt_euclidean_3D,
    visualize_gt_position_orientation,
    visualize_gt_spherical,
    visualize_equivariance_euclidean_2D,
)
from experiments.fitting.utils.ground_truth.gt_solver import (
    euclidean_2D_ffm,
    create_grid_data_euclidean_2D,
    euclidean_3D_ffm,
    create_grid_data_euclidean_3D,
    position_orientation_taichi,
    create_grid_data_position_orientation,
    spherical_agd,
    create_grid_data_spherical,
)

import math
import os


def get_gt_solver(cfg):

    if cfg.geometry.input_space == "Euclidean":

        if cfg.geometry.dim_signal == 2:
            gt_solver = {
                "grid_data": [
                    {
                        "name": "full",
                        "data": partial(
                            create_grid_data_euclidean_2D,
                            x_min=cfg.data.x_min,
                            x_max=cfg.data.x_max,
                            skip_r=cfg.eikonal.ground_truth.skip_r,
                            skip_s=cfg.eikonal.ground_truth.skip_s,
                            top_benchmark=False,
                        ),
                    },
                    {
                        "name": "top",
                        "data": partial(
                            create_grid_data_euclidean_2D,
                            x_min=cfg.data.x_min,
                            x_max=cfg.data.x_max,
                            skip_r=cfg.eikonal.ground_truth.skip_r,
                            skip_s=cfg.eikonal.ground_truth.skip_s,
                            top_benchmark=True,
                        ),
                    },
                ],
                "solver": euclidean_2D_ffm,
            }

        elif cfg.geometry.dim_signal == 3:

            gt_solver = {
                "grid_data": [
                    {
                        "name": "full",
                        "data": partial(
                            create_grid_data_euclidean_3D,
                            x_min=cfg.data.x_min,
                            x_max=cfg.data.x_max,
                            skip_r=cfg.eikonal.ground_truth.skip_r,
                            skip_s=cfg.eikonal.ground_truth.skip_s,
                        ),
                    }
                ],
                "solver": euclidean_3D_ffm,
            }

        else:
            raise ValueError()

    elif cfg.geometry.input_space == "Position_Orientation":
        gt_solver = {
            "grid_data": [
                {
                    "name": "full",
                    "data": partial(
                        create_grid_data_position_orientation,
                        x_min=cfg.data.x_min,
                        x_max=cfg.data.x_max,
                        skip_r=cfg.eikonal.ground_truth.skip_r,
                        skip_s=cfg.eikonal.ground_truth.skip_s,
                        skip_s_theta=cfg.eikonal.ground_truth.skip_s_theta,
                        theta_range=cfg.geometry.theta_range,
                    ),
                }
            ],
            "solver": partial(
                position_orientation_taichi,
                device=cfg.eikonal.ground_truth.device,
                xi=cfg.geometry.metric.xi,
                epsilon=cfg.geometry.metric.epsilon,
                sub_riem=cfg.eikonal.ground_truth.sub_riem,
                n_max=cfg.eikonal.ground_truth.n_max,
                n_max_initialisation=cfg.eikonal.ground_truth.n_max_initialisation,
                n_check=cfg.eikonal.ground_truth.n_check,
                n_check_initialisation=cfg.eikonal.ground_truth.n_check_initialisation,
                tol=cfg.eikonal.ground_truth.tol,
                initial_condition=cfg.eikonal.ground_truth.initial_condition,
            ),
        }

    elif cfg.geometry.input_space == "Spherical":
        gt_solver = {
            "grid_data": [
                {
                    "name": "full",
                    "data": partial(
                        create_grid_data_spherical,
                        x_min=cfg.data.x_min,
                        x_max=cfg.data.x_max,
                        skip_r=cfg.eikonal.ground_truth.skip_r,
                        skip_s=cfg.eikonal.ground_truth.skip_s,
                    ),
                },
            ],
            "solver": spherical_agd,
        }

    else:

        raise ValueError()

    return gt_solver


def get_recon_visualization(cfg, vmin, vmax):

    if cfg.geometry.input_space == "Euclidean":
        if cfg.geometry.dim_signal == 2:

            return partial(
                visualize_reconstructions_euclidean_2D,
                max_num_visualized_rec=cfg.visualization.max_num_visualized_rec,
                max_pairs_plot=cfg.visualization.max_pairs_plot,
                x_min=cfg.data.x_min,
                x_max=cfg.data.x_max,
                vmin=vmin,
                vmax=vmax,
            )
        elif cfg.geometry.dim_signal == 3:
            return partial(
                visualize_reconstructions_euclidean_3D,
                max_num_visualized_rec=cfg.visualization.max_num_visualized_rec,
                max_pairs_plot=cfg.visualization.max_pairs_plot,
                x_min=cfg.data.x_min,
                x_max=cfg.data.x_max,
                vmin=vmin,
                vmax=vmax,
            )
        else:
            raise ValueError()

    elif cfg.geometry.input_space == "Position_Orientation":

        theta_range = (
            [
                0.0,
                2.0 * math.pi,
            ]
            if cfg.geometry.theta_range == "zero"
            else [-math.pi, math.pi]
        )
        return partial(
            visualize_reconstructions_euclidean_3D,
            max_num_visualized_rec=cfg.visualization.max_num_visualized_rec,
            max_pairs_plot=cfg.visualization.max_pairs_plot,
            x_min=cfg.data.x_min + [theta_range[0]],
            x_max=cfg.data.x_max + [theta_range[1]],
            vmin=vmin - 1e-3,
            vmax=vmax + 1e-3,
            label_z="θ",  # theta unicode
        )
    elif cfg.geometry.input_space == "Spherical":
        return partial(
            visualize_reconstructions_spherical,
            max_num_visualized_rec=cfg.visualization.max_num_visualized_rec,
            max_pairs_plot=cfg.visualization.max_pairs_plot,
            x_min=cfg.data.x_min,
            x_max=cfg.data.x_max,
            vmin=vmin,
            vmax=vmax,
        )

    else:
        raise ValueError()


def get_gt_visualization(cfg, vmin, vmax, final=False):

    final_path = None
    if final:
        final_path = f"./experiments/fitting/figures/{cfg.data.base_dataset.name}"
        os.makedirs(
            final_path + "/val",
            exist_ok=True,
        )
        os.makedirs(
            final_path + "/test",
            exist_ok=True,
        )

    if cfg.geometry.input_space == "Euclidean":
        if cfg.geometry.dim_signal == 2:

            return partial(
                visualize_gt_euclidean_2D, vmin=vmin, vmax=vmax, final_path=final_path
            )
        elif cfg.geometry.dim_signal == 3:

            return partial(visualize_gt_euclidean_3D, vmin=vmin, vmax=vmax)
        else:
            raise ValueError()
    elif cfg.geometry.input_space == "Position_Orientation":
        return partial(visualize_gt_position_orientation, vmin=vmin, vmax=vmax)

    elif cfg.geometry.input_space == "Spherical":
        return partial(
            visualize_gt_spherical,
            vmin=vmin,
            vmax=vmax,
            final_path=final_path,
        )

    else:
        raise ValueError()


def get_equiv_visualization(cfg, vmin, vmax, final=False):

    final_path = None
    if final:
        final_path = f"./experiments/fitting/figures/{cfg.data.base_dataset.name}"
        os.makedirs(
            final_path + "/val",
            exist_ok=True,
        )
        os.makedirs(
            final_path + "/test",
            exist_ok=True,
        )

    if cfg.geometry.input_space == "Euclidean":
        if cfg.geometry.dim_signal == 2:

            return partial(
                visualize_equivariance_euclidean_2D,
                x_min=cfg.data.x_min,
                x_max=cfg.data.x_max,
                vmin=vmin,
                vmax=vmax,
                final_path=final_path,
            )
        else:

            raise ValueError()

    else:

        raise ValueError()
