import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# ==========================
# 0) Parameters
# ==========================
eta = 1.23  # parameter in the analytical expression

# ==========================
# 1) Plot function (APL-like style)
# ==========================
def plot_apl_style(filename: str, eta_val: float):
    # Global style (thin, journal-like)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "mathtext.fontset": "stix",

        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,
        "xtick.major.size": 3,
        "ytick.major.size": 3,

        "axes.linewidth": 0.8,
        "axes.edgecolor": "0.3",
        "lines.linewidth": 1.2,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "svg.fonttype": "none",
    })

    print(f"Loading file: {filename} ...")

    try:
        df = pd.read_csv(filename, sep=r"\s+", comment="#", engine="python")
    except Exception as e:
        raise ValueError(f"Failed to read file: {e}")

    print("Preview (first 3 rows):")
    print(df.head(3))

    target_x = "beta"
    target_y = "tan_val"

    # Require expected column names
    if target_x not in df.columns or target_y not in df.columns:
        raise ValueError(
            f"Column mismatch.\n"
            f"Expected: '{target_x}' and '{target_y}'\n"
            f"Found: {list(df.columns)}\n"
        )

    x_data = df[target_x].values
    y_data = df[target_y].values

    # Analytical curve: tan(theta) = 1 / (eta * beta)
    x_smooth = np.linspace(x_data.min(), x_data.max(), 200)
    y_smooth = 1.0 / (eta_val * x_smooth)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    # Simulation: square markers
    ax.plot(
        x_data, y_data,
        linestyle="none",
        marker="s",
        markersize=5,
        markerfacecolor="tab:blue",
        markeredgecolor="tab:blue",
        label="Simulation",
    )

    # Theory: solid line
    ax.plot(
        x_smooth, y_smooth,
        color="tab:red",
        linestyle="-",
        linewidth=1.2,
        label="Analytical",
    )

    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\tan \theta_{\mathrm{sk}}$")

    # Legend styling
    leg = ax.legend(frameon=True, loc="best")
    leg.get_frame().set_edgecolor("0.85")
    leg.get_frame().set_alpha(0.9)

    ax.minorticks_on()
    plt.tight_layout()

    outname = "apl_hall_angle_result.svg"
    plt.savefig(outname, format="svg", bbox_inches="tight")
    print(f"Saved figure: {outname}")
    plt.show()


# ==========================
# 2) Entry point
# ==========================
if __name__ == "__main__":
    data_filename = "Summary_Results.txt"

    if not os.path.exists(data_filename):
        raise FileNotFoundError(f"Input file not found: '{data_filename}'")

    plot_apl_style(data_filename, eta_val=eta)
