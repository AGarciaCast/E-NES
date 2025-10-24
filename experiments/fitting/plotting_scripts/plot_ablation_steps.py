import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def set_neurips_style():
    """Set the NeurIPS publication style for plots"""
    plt.rcParams.update(
        {
            # Use serif fonts - NeurIPS uses Times Roman (ptm)
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            # Font sizes
            "font.size": 14,  # Regular text
            "axes.titlesize": 14,  # Title size
            "axes.labelsize": 14,  # Axis label size
            "xtick.labelsize": 14,  # X tick label size
            "ytick.labelsize": 14,  # Y tick label size
            "legend.fontsize": 12,  # Legend font size
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


def create_combined_plots(top_view=True):
    """Create a combined figure with all datasets"""
    # Set NeurIPS style
    set_neurips_style()

    # Dataset configurations
    datasets = [
        {
            "name": "fwi_flatvel_a",
            "title": "FlatVel-A",
            "meta": {
                "time": 5.9,
                "re": 0.010653805192559958 if top_view else 0.013038022201508284,
            },
        },
        {
            "name": "fwi_flatvel_b",
            "title": "FlatVel-B",
            "meta": {
                "time": 5.9,
                "re": 0.02274198556318879 if top_view else 0.030766227990388872,
            },
        },
        {
            "name": "fwi_curvevel_a",
            "title": "CurveVel-A",
            "meta": {
                "time": 5.9,
                "re": 0.021961212512105702 if top_view else 0.02460063006728887,
            },
        },
        {
            "name": "fwi_curvevel_b",
            "title": "CurveVel-B",
            "meta": {
                "time": 5.9,
                "re": 0.0358279300481081 if top_view else 0.04977341987192631,
            },
        },
        {
            "name": "fwi_flatfault_a",
            "title": "FlatFault-A",
            "meta": {
                "time": 5.9,
                "re": 0.013719352977350354 if top_view else 0.017493647895753382,
            },
        },
        {
            "name": "fwi_flatfault_b",
            "title": "FlatFault-B",
            "meta": {
                "time": 5.9,
                "re": 0.030583238489925862 if top_view else 0.029979822300374503,
            },
        },
        {
            "name": "fwi_curvefault_a",
            "title": "CurveFault-A",
            "meta": {
                "time": 5.9,
                "re": 0.02085731126368046 if top_view else 0.02470500398427248,
            },
        },
        {
            "name": "fwi_curvefault_b",
            "title": "CurveFault-B",
            "meta": {
                "time": 5.9,
                "re": 0.03812311436980963 if top_view else 0.03823662627488375,
            },
        },
        {
            "name": "fwi_style_a",
            "title": "Style-A",
            "meta": {
                "time": 5.9,
                "re": 0.013164829714223742 if top_view else 0.013258935790508986,
            },
        },
        {
            "name": "fwi_style_b",
            "title": "Style-B",
            "meta": {
                "time": 5.9,
                "re": 0.015405030623078346 if top_view else 0.015657416768372057,
            },
        },
    ]

    # Create figure with adjusted dimensions for 2 rows of 4 columns and 1 row of 2 columns
    fig = plt.figure(figsize=(12, 9))

    # Get a representative dataset to extract steps for the legend
    rep_df = pd.read_csv(
        f"./experiments/fitting/results/results_{datasets[0]['name']}.csv"
    )
    steps = rep_df["steps"].tolist()

    # Use viridis colormap for better scientific visualization
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(steps), max(steps))

    # Create subplots in a 2 rows × 4 columns + 1 row × 2 columns layout
    for i, dataset in enumerate(datasets):
        if i < 8:  # First 8 datasets in 2 rows of 4
            row = i // 4
            col = i % 4
            ax = plt.subplot2grid((3, 4), (row, col))
        else:  # Last 2 datasets centered in the third row
            col = (i - 8) + 1  # Start from the second column (for centering)
            ax = plt.subplot2grid((3, 4), (2, col))

        # Read data
        df = pd.read_csv(f"./experiments/fitting/results/results_{dataset['name']}.csv")

        if top_view:
            re = df["test_top_re"]
        else:
            re = df["test_full_re"]

        time = df["test_neural_fit_comp_times"] / 100
        labels = df["steps"]

        # Create colors using viridis colormap
        scatter_colors = [cmap(norm(step)) for step in labels]

        # Plot autodecoding points
        sc = ax.scatter(
            time,
            re,
            c=scatter_colors,
            s=80,
            edgecolors="black",
            linewidths=0.5,
            alpha=0.9,
            marker="o",
        )

        # Plot meta-learning point
        ax.scatter(
            [dataset["meta"]["time"] / 100],
            [dataset["meta"]["re"]],
            marker="*",
            c="#D14D41",
            s=180,
            edgecolors="black",
            linewidths=0.2,
            zorder=10,
        )

        # Set labels and title
        ax.set_xlabel(r"Time ($10^2$ s)")
        ax.set_ylabel("Relative error")
        ax.set_title(dataset["title"])

        # Add grid for better readability
        ax.grid(True, linestyle="--", alpha=0.7)

        # Adjust margins
        ax.margins(x=0.1)

    # Apply tight layout first to position all subplots
    plt.tight_layout()

    # Create legend elements after subplots for proper placement

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="#D14D41",
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=18,
            label="Meta-learning",
        ),
    ]

    # Add step annotations to legend with viridis colors
    # First, add a label for the autodecoding steps
    legend_elements.append(
        Line2D([0], [0], marker=None, linestyle="none", label="Autodecoding steps:")
    )

    # Then add step values with corresponding colors
    for i, step in enumerate(steps):
        color = cmap(norm(step))
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.5,
                markersize=12,
                label=f"{step}",
            )
        )

    # Add the legend at the top of the figure with a border frame
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(legend_elements),
        frameon=True,  # Add border
        facecolor="white",  # White background
        edgecolor="black",  # Black border
        framealpha=1.0,  # Solid background
        borderpad=0.8,  # Padding inside the frame
        handletextpad=0.4,  # Space between marker and text
        columnspacing=0.8,  # Space between columns
        fontsize="small",
    )

    # Save the combined figure
    if top_view:
        plt.savefig(
            "./experiments/fitting/figures/combined_top_plots.pdf", bbox_inches="tight"
        )
    else:
        plt.savefig(
            "./experiments/fitting/figures/combined_full_plots.pdf", bbox_inches="tight"
        )

    return fig


if __name__ == "__main__":
    # Create top view combined figure
    fig_top = create_combined_plots(top_view=True)

    # Create full view combined figure
    fig_full = create_combined_plots(top_view=False)

    plt.show()
