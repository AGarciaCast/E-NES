import numpy as np
import matplotlib.pyplot as plt

# Set up publication-quality parameters
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.usetex": True,
        "axes.labelsize": 14,
        "font.size": 18,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "figure.figsize": (4, 3),  # Standard figure size for single-column journals
        "text.latex.preamble": r"\usepackage{amsmath}",
        "axes.linewidth": 0.5,
        "grid.linewidth": 0.5,
    }
)

# Create x values in the interval [-5, 5]
x = np.linspace(-5, 5, 1000)

# Calculate function values
v1 = np.exp(-(x**2))
v2 = 2 * np.exp(-((x / 2) ** 2))

# Create the plot
fig, ax = plt.subplots(layout="constrained")

# Plot the functions with appropriate line styles and colors

ax.plot(x, v1, color="#4385BE", linewidth=1.5, label=r"$v$")  # Blue color
ax.plot(
    x, v2, color="#DA702C", linewidth=1.5, label=r"$g\cdot v$"  # Orange/copper color
)

# Set the limits exactly to the interval requested
ax.set_xlim(-5, 5)
ax.set_ylim(0, 2.1)  # Allow a bit of space at the top for the legend

# Add labels with proper LaTeX formatting
# ax.set_xlabel(r"$s$")

# Add a title (not always needed in papers, but included here)

# Add a grid with light lines
ax.grid(True, linestyle="--", alpha=0.3)

# Add ticks at appropriate intervals
ax.set_xticks(np.arange(-5, 6, 1))
ax.set_yticks(np.arange(0, 2.1, 0.5))

# Add a legend with a clean, professional look
ax.legend(framealpha=0.9, edgecolor="none", loc="upper right")

# For publication, the figure size should be appropriate for the journal
# Common width is 3.5 inches for single column
fig.set_dpi(300)  # High resolution for print

# Display the plot
plt.savefig("./experiments/fitting/figures/gaussian_functions.pdf", bbox_inches="tight")
plt.show()
