import numpy as np
import matplotlib.pyplot as plt
import os
import sys  # used for exiting on missing/invalid input

# ==========================
# 0) Parameters
# ==========================
eta = 1.23
beta1 = 0.05
beta2 = 0.03

# Input data file (relative path)
data_filename = "Data_Simple_Traj/Summary_Lambda_Results.txt"

# Global style knobs
AXES_W = 0.8
TICK_W = 0.8
LINE_W = 1.2
MARKER_SZ = 5.0
MARKER_EDGE_W = 1.0

# ==========================
# 1) Matplotlib style
# ==========================
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
    "xtick.major.width": TICK_W,
    "ytick.major.width": TICK_W,

    "axes.linewidth": AXES_W,
    "axes.edgecolor": "0.3",
    "lines.linewidth": LINE_W,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "svg.fonttype": "none",
})

# ==========================
# 2) Load data (lambda, theta)
# ==========================
if os.path.exists(data_filename):
    print(f"Loading file: {data_filename} ...")
    try:
        # skip header lines: comment lines + two-line table header
        data = np.loadtxt(data_filename, comments="#", skiprows=2)
        lam = data[:, 0]            # lambda
        theta_deg_sim = data[:, 1]  # theta (deg)
    except Exception as e:
        print("Error: failed to parse the input file.")
        print(f"Original exception: {e}")
        sys.exit(1)
else:
    print(f"Error: input file not found: {data_filename}")
    sys.exit(1)

# Analytical curve
lam_fit = np.linspace(lam.min(), lam.max(), 400)
tan_theta_fit = (1.0 / eta) * (1.0 - lam_fit) / (beta1 + beta2 * lam_fit)
theta_deg_fit = np.rad2deg(np.arctan(tan_theta_fit))

# ==========================
# 3) Plot: theta_sk vs lambda
# ==========================
fig, ax = plt.subplots(figsize=(3.5, 2.8))

# Simulation: hollow square markers
ax.plot(
    lam,
    theta_deg_sim,
    linestyle="none",
    marker="s",
    markersize=MARKER_SZ,
    markerfacecolor="none",
    markeredgecolor="goldenrod",
    markeredgewidth=MARKER_EDGE_W,
    label="Simulation",
)

# Analytical: solid line
ax.plot(
    lam_fit,
    theta_deg_fit,
    color="forestgreen",
    linestyle="-",
    linewidth=LINE_W,
    label="Analytical",
)

ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"$\theta_{\mathrm{sk}}$ (deg)")

# Legend styling
leg = ax.legend(frameon=True, loc="best")
frame = leg.get_frame()
frame.set_edgecolor("0.85")
frame.set_alpha(0.9)

ax.minorticks_on()
plt.tight_layout()

outname = "lambda_hall_angle_greenyellow_strict.svg"
plt.savefig(outname, format="svg", bbox_inches="tight")
print(f"Saved figure: {outname}")

plt.show()
