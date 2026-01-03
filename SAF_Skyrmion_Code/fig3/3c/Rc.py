import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# =========================
# 1) Theory parameters
# =========================
beta1 = 0.0316
beta2 = 0.0510
eta   = 1.23          # dimensionless parameter
lam1  = 0.5           # reference λ1 for normalization

# Rc(λ1, λ2) = jc(λ2) / jc(λ1)
def Rc(l1, l2, eta, beta1, beta2):
    """Analytical ratio Rc(λ1, λ2)."""
    l2 = np.asarray(l2)
    # D(λ) = η^2(β1 + β2 λ)^2 + (1 - λ)^2
    D1 = eta**2 * (beta1 + beta2 * l1)**2 + (1.0 - l1)**2
    D2 = eta**2 * (beta1 + beta2 * l2)**2 + (1.0 - l2)**2
    return np.sqrt(D1 / D2)

# =========================
# 2) Load simulation data
# =========================
# Expected file format (two columns):
#   lambda   jc(lambda)
# Lines starting with "#" are ignored.
data_file = "jx3_lambda_0.03_0.05.txt"

print(f"Loading data file: {data_file}")
data = np.loadtxt(data_file, comments="#", usecols=(0, 1))

lam_data = data[:, 0]   # λ
jc_data  = data[:, 1]   # jc(λ)

# Normalize by jc at λ1 = 0.5 (use the closest sampled point)
idx_lam1 = np.argmin(np.abs(lam_data - lam1))
jc_lam1  = jc_data[idx_lam1]
jc_ratio = jc_data / jc_lam1

# =========================
# 3) Theory curve
# =========================
lam2 = np.linspace(lam_data.min(), lam_data.max(), 400)
Rc_vals = Rc(lam1, lam2, eta, beta1, beta2)

# =========================
# 4) Plot
# =========================
mpl.rcParams.update({
    "figure.figsize": (3.4, 2.6),
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": False,
    "ytick.right": False,
})

fig, ax = plt.subplots()

# Theory
ax.plot(lam2, Rc_vals, color="tab:green", linewidth=1.0, label="Theory")

# Simulation
ax.plot(
    lam_data, jc_ratio,
    linestyle="none",
    marker="o",
    markersize=4,
    markerfacecolor="white",
    markeredgecolor="tab:orange",
    markeredgewidth=1.0,
    label="Simulation",
)

ax.set_xlabel("λ₂")
ax.set_ylabel(r"$R_c(\lambda_1=0.5)$")
ax.set_xlim(lam_data.min(), lam_data.max())

leg = ax.legend(loc="upper right", fontsize=8, frameon=True)
leg.get_frame().set_edgecolor("0.8")
leg.get_frame().set_linewidth(0.6)

fig.tight_layout()

# Save figure
save_name = f"Rc_lambda1_{lam1:.2f}_vs_lambda2"
fig.savefig(f"{save_name}.svg", format="svg", bbox_inches="tight", dpi=600)

plt.show()
