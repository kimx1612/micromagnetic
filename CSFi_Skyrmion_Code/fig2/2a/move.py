import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# =========================
# 1) Global plotting style
# =========================
mpl.rcParams.update({
    "font.family": "sans-serif",
    "savefig.dpi": 300,
    "font.size": 16,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})

# =========================
# 2) Grid and snapshots
# =========================
NX, NY, NZ = 300, 85, 3
dx, dy = 1e-9, 1e-9  # meters

# Snapshot indices and annotations
steps = [2, 5, 20, 30]
etas  = [1.2316, 1.2321, 1.227, 1.227]
times = ["0.4 ns", "1 ns", "4 ns", "6 ns"]
assert len(steps) == len(etas) == len(times)

# =========================
# 3) Data loader
# =========================
def load_layer_mz_from_npy(step):
    """Load mz of the first layer from saved spin vector (Fortran order)."""
    fname = f"npy/m_step_{step}.npy"
    vec = np.load(fname)

    # vec is flattened spin: reshape to (3, NX, NY, NZ) using Fortran order
    arr = np.reshape(vec, (3, NX, NY, NZ), order="F")

    # z-component (index 2), first layer (index 0)
    mz = arr[2, :, :, 0]

    # transpose to image coordinates (NY, NX)
    return np.transpose(mz)

# =========================
# 4) Figure layout
# =========================
n = len(steps)

panel_width = 7.0
data_aspect = NX / NY
panel_height = panel_width / data_aspect

fig, axes = plt.subplots(
    n, 1,
    figsize=(panel_width, panel_height * n),
    sharex=True,
    sharey=True,
)

if n == 1:
    axes = [axes]

# Physical coordinates (nm) for imshow extent
x_nm = np.linspace(0, (NX - 1) * dx * 1e9, NX)
y_nm = np.linspace(0, (NY - 1) * dy * 1e9, NY)
extent = [x_nm[0], x_nm[-1], y_nm[0], y_nm[-1]]

for idx, (ax, step, eta, tstr) in enumerate(zip(axes, steps, etas, times)):
    img = load_layer_mz_from_npy(step)

    im = ax.imshow(
        img,
        cmap="coolwarm",
        origin="lower",
        extent=extent,
        vmin=-1.0,
        vmax=1.0,
        interpolation="lanczos",
        aspect="equal",
    )

    # Marker: a semi-transparent circle centered at (30, 30) nm
    circle = Circle(
        (30.0, 30.0),
        radius=3.0,
        facecolor="yellow",
        edgecolor="yellow",
        alpha=0.4,
        linewidth=1.0,
        zorder=3,
    )
    ax.add_patch(circle)

    # Panel annotations
    ax.text(
        0.98, 0.85,
        rf"$\eta = {eta:.4f}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="white",
        fontsize=16,
        fontweight="bold",
    )

    ax.text(
        0.02, 0.85,
        rf"$t = {tstr}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=16,
        fontweight="bold",
    )

    # x-label on the bottom panel only
    if idx == n - 1:
        ax.set_xlabel("X (nm)", fontsize=16)

    ax.tick_params(labelsize=12)

# Shared y-label and spacing
fig.supylabel("Y (nm)", fontsize=16, x=0.06)
fig.subplots_adjust(hspace=0.1, left=0.12, right=0.95, top=0.95, bottom=0.1)

print(f"Saving figure, Dimensions: {panel_width} x {panel_height*n:.2f} inches")
plt.savefig("skyrmion_strips_shared_Y.pdf", bbox_inches="tight")
plt.savefig("skyrmion_strips_shared_Y.svg", bbox_inches="tight", dpi=600)
plt.show()
