import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Set up matplotlib for NeurIPS style
plt.rcParams.update(
    {
        # Use serif fonts - NeurIPS uses Times Roman (ptm)
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        # Font sizes
        "font.size": 14,  # Regular text
        "axes.titlesize": 10,  # Title size
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

# Load the CSV files
# Replace with your actual file names
df_eikonal = pd.read_csv("./experiments/fitting/results/eiko_equiv.csv")
df_mse = pd.read_csv(
    "./experiments/fitting/results/mse_equiv.csv"
)  # Replace with your actual MSE data file

# Create figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.14 * 1.2, 2.79 * 1.2), dpi=100)

# Define the columns to use
x_column = "trainer/global_step"  # Same x-axis column for both datasets


# Create the left plot (Eikonal Loss)
def create_eikonal_plot(ax, df):
    y_column1 = "style b no equiv auto - train_eiko_epoch"
    y_column2 = "best style b auto - train_eiko_epoch"

    # Create the plot with two lines using the specified colors
    (line1,) = ax.plot(
        df[x_column],
        df[y_column1],
        linewidth=4,
        color="#DA702C",  # Orange/copper color
        label="No Equivariant",
    )

    (line2,) = ax.plot(
        df[x_column],
        df[y_column2],
        linewidth=4,
        color="#4385BE",  # Blue color
        label="Equivariant",
    )

    # Set y-axis limits
    ax.set_ylim(top=0.1)
    ax.set_ylim(bottom=0.035)

    # Add labels
    ax.set_xlabel(r"Step ($\times 1000$)")
    ax.set_ylabel("Eikonal Loss")

    # Add light grid for readability
    ax.grid(True, linestyle="--", alpha=0.3)

    # Format x-axis tick labels
    if df[x_column].max() > 1000:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x/1000)}"))

    # Add legend in the best location
    # ax.legend(loc="best", frameon=False)

    return ax


# Create the right plot (MSE from different DataFrame with log y-axis)
def create_mse_plot(ax, df):
    y_column1 = "style b no equiv auto - train_mse_epoch"
    y_column2 = "best style b auto - train_mse_epoch"

    # Create the plot with two lines using the specified colors (matching Eikonal plot)
    (line1,) = ax.plot(
        df[x_column],
        df[y_column1],
        linewidth=4,
        color="#DA702C",  # Orange/copper color
        label="No Equivariant",
    )

    (line2,) = ax.plot(
        df[x_column],
        df[y_column2],
        linewidth=4,
        color="#4385BE",  # Blue color
        label="Equivariant",
    )

    # Set y-axis to logarithmic scale
    ax.set_yscale("log")

    # Add labels
    ax.set_xlabel(r"Step ($\times 1000$)")
    ax.set_ylabel("MSE (log scale)")

    # Add light grid for readability
    ax.grid(True, linestyle="--", alpha=0.3)

    ax.set_ylim(top=0.1)

    # Format x-axis tick labels
    if df[x_column].max() > 1000:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x/1000)}"))

    # Add legend in the best location
    ax.legend(loc="best", frameon=False)

    return ax


# Create the plots using their respective DataFrames
create_eikonal_plot(ax1, df_eikonal)
create_mse_plot(ax2, df_mse)

# Adjust spacing between subplots
plt.tight_layout()

# Save the plot as a PDF and PNG for inclusion in LaTeX document
plt.savefig("./experiments/fitting/figures/equivariance.pdf", bbox_inches="tight")

# Display the plot
plt.show()
