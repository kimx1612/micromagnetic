import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# ============================
# 1) Global plotting style
# ============================
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
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,

    "axes.linewidth": 0.8,
    "axes.edgecolor": "0.3",
    "lines.linewidth": 1.2,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "svg.fonttype": "none",
})

# ============================
# 2) Parameters and data loading
# ============================
eta = 1.23
beta1_true = 0.05
beta2_true = 0.03

filename = "Data_Simple_Traj/Summary_Lambda_Results.txt"

# Strict file loading: exit on missing/invalid input
try:
    print(f"Loading file: {filename} ...")
    data = np.loadtxt(filename, comments="#", skiprows=2)
    lam = data[:, 0]
    tan_sim = data[:, 2]  # column 3: tan(theta)
except FileNotFoundError:
    print(f"Error: input file not found: {filename}")
    sys.exit(1)
except Exception as e:
    print(f"Error: failed to read/parse the file. Original exception: {e}")
    sys.exit(1)

print(f"Loaded {len(lam)} data points.")

# Compute Y(lambda) from simulation
Y_data = (1.0 - lam) / tan_sim

# ============================
# 3) Data filtering (fit range)
# ============================
# Exclude lambda ~ 1.0 to avoid singular behavior
mask_non_one = (np.abs(lam - 1.0) > 1e-5)

# Fit only within 0.0 <= lambda <= 0.9
mask_range = (lam >= 0.0) & (lam <= 0.9)

# Keep finite values only
final_mask = mask_non_one & mask_range & np.isfinite(Y_data)

lam_filtered = lam[final_mask]
Y_data_filtered = Y_data[final_mask]

print(f"Original points: {len(lam)}; points used for fit: {len(lam_filtered)}")
if len(lam_filtered) == 0:
    print("Error: no valid points left after filtering.")
    sys.exit(1)

# ============================
# 4) Linear fit for beta
# ============================
# Fit: Y ≈ A + B*lambda  => beta1_fit = A/eta, beta2_fit = B/eta
B_fit, A_fit = np.polyfit(lam_filtered, Y_data_filtered, 1)

beta1_fit = A_fit / eta
beta2_fit = B_fit / eta

print(f"Fit results: beta1_fit = {beta1_fit:.6f}, beta2_fit = {beta2_fit:.6f}")

# Theory line extended to lambda = 1.0
lam_fit = np.linspace(lam_filtered.min(), 1.0, 400)
Y_fit_theory = eta * (beta1_true + beta2_true * lam_fit)

# ============================
# 5) Plot: Y(lambda)
# ============================
fig, ax = plt.subplots(figsize=(3.4, 2.6))

# Data: hollow blue squares
ax.plot(
    lam_filtered,
    Y_data_filtered,
    linestyle="none",
    marker="s",
    markersize=6.0,
    markerfacecolor="none",
    markeredgecolor="royalblue",
    markeredgewidth=1.0,
    label=r"$Y_{\mathrm{data}}$",
)

# Theory: solid red line
ax.plot(
    lam_fit,
    Y_fit_theory,
    "-",
    color="crimson",
    linewidth=1.2,
    label=r"Theory",
)

ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"$Y(\lambda)$")

# Annotation: fitted parameters
param_text = (
    rf"$\beta_1^{{\mathrm{{fit}}}} = {beta1_fit:.4f}$" + "\n\n"
    rf"$\beta_2^{{\mathrm{{fit}}}} = {beta2_fit:.4f}$"
)
ax.text(
    0.60, 0.15,
    param_text,
    transform=ax.transAxes,
    fontsize=11,
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2),
)

# Legend styling
leg = ax.legend(loc="upper left", frameon=True)
leg_frame = leg.get_frame()
leg_frame.set_edgecolor("0.85")
leg_frame.set_alpha(0.9)

ax.minorticks_on()
ax.set_xlim(lam_filtered.min() - 0.05, 1.0 + 0.05)

plt.tight_layout()

print("Saving figures...")
plt.savefig("Fig_Y_Lambda_Extended_Theory.svg", format="svg", bbox_inches="tight")
plt.savefig("Fig_Y_Lambda_Extended_Theory.png", dpi=300, bbox_inches="tight")
print("Done.")

plt.show()
