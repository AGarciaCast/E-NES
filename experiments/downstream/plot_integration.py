import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def plot_curve_sphere_2d(
    points,
    vel,
    x_min=None,
    x_max=None,
    vmin=0.0,
    vmax=1.0,
    wrap_threshold=None,
    title="",
):
    # Set paper-ready matplotlib parameters
    plt.rcParams.update(
        {
            # Use serif fonts - NeurIPS uses Times Roman (ptm)
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            # Font sizes - increased
            "font.size": 24,  # Regular text
            "axes.titlesize": 24,  # Title size
            "axes.labelsize": 26,  # Axis label size
            "xtick.labelsize": 18,  # X tick label size
            "ytick.labelsize": 18,  # Y tick label size
            "legend.fontsize": 16,  # Legend font size
            # Line widths
            "axes.linewidth": 0.5,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.0,
            "lines.markersize": 3,
            # Clean style for academic publications
            "axes.grid": False,
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "grid.color": "#CCCCCC",
            "grid.linestyle": "--",
            # Legend settings
            "legend.frameon": False,
            "legend.numpoints": 1,
            "legend.handlelength": 2,
            # Use TrueType fonts for better PDF output
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    points = [tuple(row) for row in points.tolist()]

    start = points[0]
    goal = points[-1]

    nx, ny = vel.shape
    xmin, ymin = x_min if x_min is not None else [0, 0]
    xmax, ymax = x_max if x_max is not None else [2 * np.pi, np.pi]
    wrap_threshold = wrap_threshold if wrap_threshold is not None else np.pi

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)

    # Create figure with proper aspect ratio (2:1 for 2π:π)
    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot velocity field (inverted y-axis)
    im = ax.imshow(
        vel.T,
        extent=[xmin, xmax, ymax, ymin],  # Inverted for y-axis
        origin="upper",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        alpha=0.8,
        aspect="equal",  # Keep equal aspect for proper proportions
    )

    # Segment the path at periodic boundary crossings
    def segment_path(points, wrap_threshold):
        if len(points) < 2:
            return [points]

        segments = []
        current_segment = [points[0]]

        for i in range(1, len(points)):
            x_diff = abs(points[i][0] - points[i - 1][0])

            if x_diff > wrap_threshold:
                segments.append(current_segment)
                current_segment = [points[i]]
            else:
                current_segment.append(points[i])

        if current_segment:
            segments.append(current_segment)

        return segments

    path_segments = segment_path(points, wrap_threshold)

    # Plot path segments with thicker lines
    for segment in path_segments:
        if len(segment) > 1:
            seg_array = np.array(segment)
            ax.plot(
                seg_array[:, 0],
                seg_array[:, 1],
                "-",
                color="tab:red",
                linewidth=5,
                zorder=3,
            )

    # Plot start marker (diamond) and goal marker (circle) - thicker
    ax.plot(
        start[0],
        start[1],
        "D",
        color="tab:red",
        markersize=16,
        markeredgecolor="black",
        markeredgewidth=2.5,
        zorder=4,
    )
    ax.plot(
        goal[0],
        goal[1],
        "o",
        color="tab:red",
        markersize=20,
        markeredgecolor="black",
        markeredgewidth=2.5,
        zorder=4,
    )

    # Labels
    ax.set_xlabel("θ")
    ax.set_ylabel("φ")

    # Remove whitespace by setting tight limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)  # Inverted for y-axis

    # add title
    ax.set_title(f"Polar Coords: {title}", pad=15)
    plt.tight_layout(pad=0)
    #  Remove margins
    ax.margins(0)
    # add small space for title
    plt.subplots_adjust(top=0.85)

    # Apply tight layout with no padding

    return fig


import plotly.graph_objects as go


def plot_sphere_3d_with_path(
    vel, points, x_min=None, x_max=None, vmin=None, vmax=None, colorbar=True
):
    """Plot velocity field on a 3D sphere with path overlay - camera ready version."""

    nx, ny = vel.shape
    xmin, ymin = x_min if x_min is not None else [0, 0]
    xmax, ymax = x_max if x_max is not None else [2 * np.pi, np.pi]

    # Compute vmin/vmax from data if not provided
    if vmin is None:
        vmin = np.min(vel)
    if vmax is None:
        vmax = np.max(vel)

    # Convert JAX arrays to Python floats
    vmin = float(vmin)
    vmax = float(vmax)

    print(f"Color scale: vmin={vmin:.4f}, vmax={vmax:.4f}")

    # Create grid
    theta = np.linspace(xmin, xmax, nx)
    phi = np.linspace(ymin, ymax, ny)
    THETA, PHI = np.meshgrid(theta, phi, indexing="ij")

    # Transform to 3D Cartesian coordinates (sphere)
    X = np.cos(THETA) * np.sin(PHI)
    Y = np.sin(THETA) * np.sin(PHI)
    Z = np.cos(PHI)

    # Convert vel to numpy if it's a JAX array
    vel_np = np.array(vel)

    # Transform path points to 3D - elevate above surface
    points_array = np.array(points)
    path_theta = points_array[:, 0]
    path_phi = points_array[:, 1]

    path_elevation = 1.005  # 2.5% above surface (more than wireframe)
    path_x = path_elevation * np.cos(path_theta) * np.sin(path_phi)
    path_y = path_elevation * np.sin(path_theta) * np.sin(path_phi)
    path_z = path_elevation * np.cos(path_phi)

    # Create the figure
    fig = go.Figure()

    # Add soft, realistic drop shadow
    shadow_theta = np.linspace(xmin, xmax, 60)
    shadow_phi = np.linspace(ymin, ymax, 60)
    SHADOW_THETA, SHADOW_PHI = np.meshgrid(shadow_theta, shadow_phi, indexing="ij")

    shadow_scale = 1.0
    X_shadow = shadow_scale * np.cos(SHADOW_THETA) * np.sin(SHADOW_PHI)
    Y_shadow = shadow_scale * np.sin(SHADOW_THETA) * np.sin(SHADOW_PHI)
    Z_shadow = -1.05 * np.ones_like(X_shadow)

    # Soft Gaussian falloff
    r_shadow = np.sqrt(X_shadow**2 + Y_shadow**2)
    r_max = 1.0
    shadow_intensity = np.exp(-2.5 * (r_shadow / r_max) ** 2)
    shadow_intensity = shadow_intensity * (0.3 + 0.7 * np.sin(SHADOW_PHI))

    fig.add_trace(
        go.Surface(
            x=X_shadow,
            y=Y_shadow,
            z=Z_shadow,
            surfacecolor=shadow_intensity,
            colorscale=[
                [0, "rgba(0,0,0,0)"],
                [0.3, "rgba(0,0,0,0.05)"],
                [0.6, "rgba(0,0,0,0.15)"],
                [1, "rgba(0,0,0,0.35)"],
            ],
            showscale=False,
            name="Shadow",
            hoverinfo="skip",
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0),
            contours=dict(
                x=dict(highlight=False),
                y=dict(highlight=False),
                z=dict(highlight=False),
            ),
        )
    )

    if colorbar:
        colorbar_dict = dict(
            title=dict(
                text="Velocity Field",
                side="right",
                font=dict(size=35, family="serif", color="black"),
            ),
            thickness=35,  # Thicker colorbar
            len=0.5,  # Match matplotlib's shrink=0.5
            x=0.8,  # Much closer to the sphere
            y=0.5,
            yanchor="middle",
            tickfont=dict(size=25, family="serif", color="black"),
            outlinewidth=1.5,
            outlinecolor="black",
            lenmode="fraction",
            tickwidth=2,
            tickcolor="black",
            ticklen=8,
        )

    # Add main velocity surface with enhanced colorbar
    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            surfacecolor=vel_np,
            colorscale="Viridis",
            cmin=vmin,
            cmax=vmax,
            colorbar=colorbar_dict if colorbar else None,
            name="Velocity Field",
            showscale=colorbar,
        )
    )

    # Add wireframe ELEVATED significantly above the sphere
    wireframe_elevation = 1.001  # 1.5% above sphere surface
    wireframe_density = 12
    theta_wire = np.linspace(xmin, xmax, wireframe_density)
    phi_wire = np.linspace(ymin, ymax, wireframe_density)

    # Longitude lines - elevated and high resolution
    for t in theta_wire:
        phi_line = np.linspace(ymin, ymax, 200)
        x_line = wireframe_elevation * np.cos(t) * np.sin(phi_line)
        y_line = wireframe_elevation * np.sin(t) * np.sin(phi_line)
        z_line = wireframe_elevation * np.cos(phi_line)
        fig.add_trace(
            go.Scatter3d(
                x=x_line,
                y=y_line,
                z=z_line,
                mode="lines",
                line=dict(color="rgba(255, 255, 255, 0.01)", width=5.5),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Latitude lines - elevated and high resolution
    for p in phi_wire:
        theta_line = np.linspace(xmin, xmax, 200)
        x_line = wireframe_elevation * np.cos(theta_line) * np.sin(p)
        y_line = wireframe_elevation * np.sin(theta_line) * np.sin(p)
        z_line = wireframe_elevation * np.cos(p) * np.ones_like(theta_line)
        fig.add_trace(
            go.Scatter3d(
                x=x_line,
                y=y_line,
                z=z_line,
                mode="lines",
                line=dict(color="rgba(255, 255, 255, 0.01)", width=5.5),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Add path - RED color matching 2D plot
    fig.add_trace(
        go.Scatter3d(
            x=path_x,
            y=path_y,
            z=path_z,
            mode="lines",
            line=dict(color="#d62728", width=40),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Add start point - STAR marker with #D14D41 color matching 2D plot
    fig.add_trace(
        go.Scatter3d(
            x=[path_x[0]],
            y=[path_y[0]],
            z=[path_z[0]],
            mode="markers",
            marker=dict(
                size=25,
                color="#d62728",
                symbol="diamond",
                line=dict(color="black", width=50),
            ),
            name="Start",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Add goal point - CIRCLE marker with #D14D41 color matching 2D plot
    fig.add_trace(
        go.Scatter3d(
            x=[path_x[-1]],
            y=[path_y[-1]],
            z=[path_z[-1]],
            mode="markers",
            marker=dict(
                size=25,
                color="#d62728",
                symbol="circle",
                line=dict(color="black", width=50),
            ),
            name="Goal",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=[-1.3, 1.3]),
            yaxis=dict(visible=False, range=[-1.3, 1.3]),
            zaxis=dict(visible=False, range=[-1.3, 1.3]),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2), center=dict(x=0, y=0, z=0)),
            bgcolor="white",
        ),
        width=1000,
        height=1000,
        paper_bgcolor="white",
        showlegend=False,
        margin=dict(l=0, r=100, t=0, b=0),
        font=dict(family="serif", size=16),
    )

    return fig


# Function to save plot with current camera view
def save_plot_to_pdf(fig, filename="sphere_plot.pdf"):
    """
    Save the current view of the plot to PDF.

    First display the plot interactively with fig.show(),
    rotate to your desired view, then call this function
    to save that exact view to PDF.

    Note: Requires kaleido package. Install with: pip install kaleido
    """
    fig.write_image(filename, format="pdf", width=1000, height=1000, scale=2)
    print(f"Saved to {filename}")


if __name__ == "__main__":
    start = [1.5, 1]
    goal = [4, 2]
    radius = 1.0
    marker_radius_factor = 1.15  # Markers placed at radius * this factor
    wandb_id = "y10qqb2n"
    idx_vel = 7
    # wandb_id = "0e1cqt34"
    # idx_vel = 56

    # Load cached data
    cache_dir = "/experiments/downstream/results"
    cache_file = os.path.join(
        cache_dir,
        f"path_{wandb_id}_vel_{idx_vel}_start_{start[0]}_{start[1]}_goal_{goal[0]}_{goal[1]}.pkl",
    )

    if not os.path.exists(cache_file):
        raise FileNotFoundError(
            f"Cache file not found: {cache_file}\n"
            "Please run gradient_int_sphere.py first to generate the cache."
        )

    print(f"Loading cached data from {cache_file}...")
    with open(cache_file, "rb") as f:
        cached_data = pickle.load(f)

    points = cached_data["points"]
    vel = cached_data["vel"]
    vmin = cached_data["vmin"]
    vmax = cached_data["vmax"]
    x_min = cached_data["x_min"]
    x_max = cached_data["x_max"]

    print(f"Loaded path with {len(points)} points")
    print(f"Velocity field shape: {vel.shape}")

    fig = plot_curve_sphere_2d(
        points,
        vel,
        x_min=x_min,
        x_max=x_max,
        vmin=vmin,
        vmax=vmax,
        title="Gaussian Obstacle Vel.",
    )

    fig.savefig(
        "/experiments/downstream/figures/sphere_2d_gauss_vel.pdf",
        format="pdf",
        dpi=300,
        bbox_inches="tight",
    )

    fig2 = plot_sphere_3d_with_path(
        vel, points, x_min=x_min, x_max=x_max, vmin=vmin, vmax=vmax
    )
    # fig2.show()

    # Step 4: Update the figure with your chosen camera angle
    fig2.update_layout(
        scene_camera=dict(
            eye=dict(x=-1.7, y=1, z=0),  # Replace with your values
            center=dict(x=0, y=0, z=0),
        )
    )

    # Step 5: Save to PDF
    save_plot_to_pdf(fig2, "/experiments/downstream/figures/my_sphere_view_gauss.pdf")

    # --------------------------------#
    wandb_id = "0e1cqt34"
    idx_vel = 56

    cache_file = os.path.join(
        cache_dir,
        f"path_{wandb_id}_vel_{idx_vel}_start_{start[0]}_{start[1]}_goal_{goal[0]}_{goal[1]}.pkl",
    )

    if not os.path.exists(cache_file):
        raise FileNotFoundError(
            f"Cache file not found: {cache_file}\n"
            "Please run gradient_int_sphere.py first to generate the cache."
        )

    print(f"Loading cached data from {cache_file}...")
    with open(cache_file, "rb") as f:
        cached_data = pickle.load(f)

    points = cached_data["points"]
    vel = cached_data["vel"]
    vel = np.full_like(vel, (vmax + vmin) / 2)

    x_min = cached_data["x_min"]
    x_max = cached_data["x_max"]

    print(f"Loaded path with {len(points)} points")
    print(f"Velocity field shape: {vel.shape}")

    fig = plot_curve_sphere_2d(
        points,
        vel,
        x_min=x_min,
        x_max=x_max,
        vmin=vmin,
        vmax=vmax,
        title="Constant Velocity Field",
    )

    fig.savefig(
        "/experiments/downstream/figures/sphere_2d_constant_vel.pdf",
        format="pdf",
        dpi=300,
        bbox_inches="tight",
    )

    fig2 = plot_sphere_3d_with_path(
        vel, points, x_min=x_min, x_max=x_max, vmin=vmin, vmax=vmax, colorbar=False
    )
    # Step 4: Update the figure with your chosen camera angle
    fig2.update_layout(
        scene_camera=dict(
            eye=dict(x=-1.7, y=1, z=0),  # Replace with your values
            center=dict(x=0, y=0, z=0),
        )
    )

    # Step 5: Save to PDF
    save_plot_to_pdf(
        fig2, "/experiments/downstream/figures/my_sphere_view_constant.pdf"
    )
