import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


# Set up matplotlib for NeurIPS style
plt.rcParams.update(
    {
        # Use serif fonts - NeurIPS uses Times Roman (ptm)
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        # Font sizes
        "font.size": 18,  # Regular text
        "axes.titlesize": 14,  # Title size
        "axes.labelsize": 18,  # Axis label size
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

df = pd.read_csv("./experiments/fitting/results/Neurips_enf_eikonal_3D_results.csv")

# Set all the columns to be float except for the first one (method)
for col in df.columns[2:]:
    df[col] = df[col].str.replace(",", ".").astype(float)

# Seperate meta from ad
df_ad = df[df["method"] == "AD"]
df_meta = df[df["method"] == "META"]
df_ad_fmm = df_ad["test_full_fmm_comp_times"]
df_meta_fmm = df_meta["test_full_fmm_comp_times"]

# Average the fmm values by adding df_add_fmm and df_meta_fmm and putting it back in a Series
df_fmm = (df_ad_fmm.values + df_meta_fmm.values) / 2
df_fmm = pd.Series(df_fmm, name="test_full_fmm_comp_times")

# Create comp time df for every grid size and make sure df_ad and df_meta are the same length
# as df_fmm by dropping the index

df_comp_time = pd.DataFrame(
    {
        "grid_size": df["grid-size"].iloc[:5],
        "test_full_fmm_comp_times": df_fmm,
        "test_neural_fit_comp_times_ad_fit": df_ad["test_neural_fit_comp_times"]
        .dropna()
        .values,
        "test_neural_fit_comp_times_meta_fit": df_meta["test_neural_fit_comp_times"]
        .dropna()
        .values,
        "test_neural_fit_comp_times_ad_inference": df_ad[
            "test_full_neural_inf_comp_times"
        ]
        .dropna()
        .values,
        "test_neural_fit_comp_times_meta_inference": df_meta[
            "test_full_neural_inf_comp_times"
        ]
        .dropna()
        .values,
    }
)


# Create a bar plot with a 3 bars for every row. One bar for 'test_full_fmm_comp_times', and than two bars for meta and ad. However, the bars for meta and ad consist of two seperate
# stacked bars consisting of 'test_neural_fit_comp_times_ad_fit' and 'test_neural_fit_comp_times_ad_inference' for ad and 'test_neural_fit_comp_times_meta_fit' and 'test_neural_fit_comp_times_meta_inference' for meta

fig, ax = plt.subplots(figsize=(8, 5))
# Set the bar width
bar_width = 0.25
# Set the x locations for the bars
x = df_comp_time["grid_size"].astype(str)
x = range(len(x))
# Set the bar locations
bar1 = [i - bar_width for i in x]
bar2 = x
bar3 = [i + bar_width for i in x]
# Set the bar colors
colors = ["#4385BE", "#DA702C", "#879A39", "#D14D41", "#8B7EC8", "#8c564b"]
# Set the bar labels
labels = [
    "FMM",
    "AD Fit",
    "AD Inference",
    "META Fit",
    "META Inference",
]
# Set the bar values
values = [
    df_comp_time["test_full_fmm_comp_times"],
    df_comp_time["test_neural_fit_comp_times_ad_fit"],
    df_comp_time["test_neural_fit_comp_times_ad_inference"],
    df_comp_time["test_neural_fit_comp_times_meta_fit"],
    df_comp_time["test_neural_fit_comp_times_meta_inference"],
]
# Create the bars
ax.bar(bar1, values[0], width=bar_width, label=labels[0], color=colors[0])
ax.bar(bar2, values[2], width=bar_width, label=labels[2], color=colors[2])
ax.bar(
    bar2, values[1], width=bar_width, label=labels[1], color=colors[1], bottom=values[2]
)
ax.bar(bar3, values[4], width=bar_width, label=labels[4], color=colors[4])
ax.bar(
    bar3, values[3], width=bar_width, label=labels[3], color=colors[3], bottom=values[4]
)
# Set the x ticks and labels
ax.set_xticks(x)
ax.set_xticklabels(df_comp_time["grid_size"].astype(str))
# Set the y label
ax.set_ylabel(r"Computation Time ($\times 10^{3}$ s)")
ax.set_xlabel(r"Source points grid size")
# Set the title
# ax.set_title("Computation Time for FMM and Neural Network")
# Set the legend
ax.legend(loc="upper left", frameon=True)


# Set the y axis formatter to scientific notation
# def scientific(x, pos):
#     return f"{x:.0e}" if x != 0 else "0"
def scientific(x, pos):
    if x == 0:
        return "0"
    coeff = x / (10 ** int(np.floor(np.log10(abs(x)))))
    exp = int(np.floor(np.log10(abs(x))))
    return r"${:.0f} \times 10^{{{}}}$".format(coeff, exp)


def scientific(x, pos):
    if x == 0:
        return "0"
    coeff = x / (10 ** int(np.floor(np.log10(abs(x)))))
    exp = int(np.floor(np.log10(abs(x))))
    return r"${:.0f}$".format(coeff)


formatter = FuncFormatter(scientific)
ax.yaxis.set_major_formatter(formatter)
# Set the grid
ax.grid(axis="y", linestyle="--", alpha=0.7)
# Set the x axis limits
ax.set_xlim(-0.5, len(x) - 0.5)
# Set the y axis limits
ax.set_ylim(0, df_comp_time["test_full_fmm_comp_times"].max() * 1.2)
# Set the y axis ticks
# ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
# Set the x axis ticks
# ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
# Set the x axis tick labels
# ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: df_comp_time["grid_size"].astype(str)[int(x)]))
# Set the y axis tick labels
# ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{x:.0e}" if x != 0 else "0"))
# Show the plot
plt.tight_layout()
plt.savefig("./experiments/fitting/figures/3d_comp_time_plot.pdf", bbox_inches="tight")

plt.close()

# Now we want to make a performance  plot with 2 lines per metric; one for ad and one for meta.
# We want to create lines for the following metrics: test_full_re, test_full_rmae,

fig, ax = plt.subplots(figsize=(8, 5))
# Set the x locations for the lines
x = df_comp_time["grid_size"].astype(str)
# Set the line colors
colors = ["#4385BE", "#ff7f0e", "#879A39", "#d62728"]
# Set the line labels
labels = [
    "AD RE",
    "AD RMAE",
    "META RE",
    "META RMAE",
]
# Set the line values
values = [
    df_ad["test_full_re"].dropna().values,
    df_ad["test_full_rmae"].dropna().values,
    df_meta["test_full_re"].dropna().values,
    df_meta["test_full_rmae"].dropna().values,
]
# Create the lines
ax.plot(
    x,
    values[0],
    label=labels[0],
    color=colors[0],
    marker="o",
    markersize=12,
    linewidth=5,
)
ax.plot(
    x,
    values[1],
    label=labels[1],
    color=colors[0],
    linestyle="--",
    marker="s",
    markersize=12,
    linewidth=5,
)
ax.plot(
    x,
    values[2],
    label=labels[2],
    color=colors[2],
    marker="o",
    markersize=12,
    linewidth=5,
)
ax.plot(
    x,
    values[3],
    label=labels[3],
    color=colors[2],
    linestyle="--",
    marker="s",
    markersize=12,
    linewidth=5,
)
# Set the x ticks and labels
ax.set_xticks(x)
ax.set_xticklabels(df_comp_time["grid_size"].astype(str))
# Set the y label
ax.set_ylabel(r"Error ($\times 10^{-2}$)")
ax.set_xlabel(r"Source points grid size")
# Set the title
# ax.set_title("RE and RMAE for FMM and Neural Network")
# Set the legend
ax.legend(loc="upper left", frameon=True)


# Set the y axis formatter to scientific notation
def scientific(x, pos):
    if x == 0:
        return "0"
    coeff = x / (10 ** int(np.floor(np.log10(abs(x)))))
    exp = int(np.floor(np.log10(abs(x))))
    return r"${:.0f} \times 10^{{{}}}$".format(coeff, exp)


def scientific(x, pos):
    if x == 0:
        return "0"
    coeff = x / (10 ** int(np.floor(np.log10(abs(x)))))
    exp = int(np.floor(np.log10(abs(x))))
    return r"${:.0f}$".format(coeff)


formatter = FuncFormatter(scientific)
ax.yaxis.set_major_formatter(formatter)

# Set the grid
ax.grid(axis="y", linestyle="--", alpha=0.7)
# Set the x axis limits
ax.set_xlim(-0.5, len(x) - 0.5)
# Set the y axis limits
ax.set_ylim(0.01, df["test_full_re"].max() * 1.6)
# reduce number of ticks

# Show the plot
plt.tight_layout()

plt.savefig(
    "./experiments/fitting/figures/3d_performance_plot.pdf", bbox_inches="tight"
)
