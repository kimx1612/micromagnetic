import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (required to register 3D projection)
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch

# ================== CONFIG ==================
# Each entry: (filename, legend label, Rd in nm)
data_files = [
    ("XY_Rd_7_lambda_0.05.txt", "Rd = 7 nm", 7.0),
    ("XY_Rd_5_lambda_0.05.txt", "Rd = 5 nm", 5.0),
    ("XY_Rd_3_lambda_0.05.txt", "Rd = 3 nm", 3.0),
]

# Output filenames
output_3d_png = "Energy_Well_3D_LargeFont.png"
output_3d_svg = "Energy_Well_3D_LargeFont.svg"
output_2d_png = "Energy_Well_Profile_2D_LargeFont.png"
output_2d_svg = "Energy_Well_Profile_2D_LargeFont.svg"

# Geometric center (nm) used to map radial profile into (X,Y) surface
center_x_nm = 100.0
center_y_nm = 80.0

# Colormap for the 3D surface + wireframe colors for different Rd
cmap_name = "cividis"
curve_colors = ["tab:orange", "tab:blue", "tab:red"]

# Font sizes
FS_TITLE  = 18   # title
FS_LABEL  = 15   # axis labels
FS_TICK   = 13   # tick labels
FS_LEGEND = 13   # legend

def load_profile(filename):
    """
    Load one trajectory/profile file and return a sorted radial energy profile.

    Input file columns (whitespace-separated):
      Time  Rx  Ry  Energy
    Rx/Ry are assumed in meters; output radius is in nm.
    Energy is converted to meV relative to the first row.
    """
    df = pd.read_csv(
        filename,
        sep=r"\s+",
        comment="#",
        names=["Time", "Rx", "Ry", "Energy"],
        engine="python",
    )

    # Reference point: first guiding-center position
    rx_0 = df["Rx"].iloc[0]
    ry_0 = df["Ry"].iloc[0]

    # Radius r in nm
    r_data = np.sqrt((df["Rx"] - rx_0) ** 2 + (df["Ry"] - ry_0) ** 2) * 1e9

    # Energy difference to meV
    J_TO_MEV = 1000.0 / 1.60217663e-19
    E_0 = df["Energy"].iloc[0]
    z_data = (df["Energy"] - E_0) * J_TO_MEV

    # Sort by radius for clean profiles
    sorted_indices = np.argsort(r_data)
    return r_data[sorted_indices], z_data[sorted_indices]


def main():
    # Run from the script directory so relative paths work
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    # Load all available profiles
    profiles = []
    for (fname, label, Rd), color in zip(data_files, curve_colors):
        if not os.path.exists(fname):
            print(f"Warning: {fname} not found, skip.")
            continue
        print(f"Loading {fname} ...")
        r, z = load_profile(fname)
        profiles.append({"label": label, "r": r, "z": z, "color": color})

    if not profiles:
        return

    # ==========================================
    # FIGURE 1: 3D Potential Well
    # ==========================================
    fig3d = plt.figure(figsize=(7, 6))
    ax3d = fig3d.add_subplot(111, projection="3d")

    theta = np.linspace(0.0, 2.0 * np.pi, 100)
    main_prof = profiles[0]  # first profile used as the surface

    # Surface from the main profile (r,z) revolved around the center
    r0 = main_prof["r"]
    z0 = main_prof["z"]
    R0, THETA0 = np.meshgrid(r0, theta)

    X0 = center_x_nm + R0 * np.cos(THETA0)
    Y0 = center_y_nm + R0 * np.sin(THETA0)
    Z0 = np.tile(z0, (len(theta), 1))

    ax3d.plot_surface(X0, Y0, Z0, cmap=cmap_name, linewidth=0, antialiased=True, alpha=0.85)
    ax3d.plot_wireframe(X0, Y0, Z0, rstride=6, cstride=6, color=main_prof["color"], linewidth=0.7)

    # Additional wireframes (other Rd)
    wire_patches = []
    for prof in profiles:
        if prof is main_prof:
            continue
        r, z, color = prof["r"], prof["z"], prof["color"]
        R, THETA = np.meshgrid(r, theta)
        X = center_x_nm + R * np.cos(THETA)
        Y = center_y_nm + R * np.sin(THETA)
        Z = np.tile(z, (len(theta), 1))
        ax3d.plot_wireframe(X, Y, Z, rstride=6, cstride=6, color=color, linewidth=0.7)
        wire_patches.append(Patch(color=color, label=prof["label"]))

    # 3D labels and view
    ax3d.set_xlabel("X (nm)", fontsize=FS_LABEL, labelpad=10)
    ax3d.set_ylabel("Y (nm)", fontsize=FS_LABEL, labelpad=10)
    ax3d.set_zlabel("dE (meV)", fontsize=FS_LABEL, labelpad=10)
    ax3d.set_title("Energy potential well (3D)", fontsize=FS_TITLE, pad=15)

    ax3d.tick_params(axis="both", which="major", labelsize=FS_TICK)
    ax3d.view_init(elev=30, azim=45)
    ax3d.zaxis.set_major_locator(MaxNLocator(5))
    ax3d.grid(False)

    # Remove pane fill for a clean look
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False

    legend_handles = [Patch(color=main_prof["color"], label=main_prof["label"])] + wire_patches
    ax3d.legend(handles=legend_handles, loc="upper left", fontsize=FS_LEGEND)

    plt.tight_layout()
    fig3d.savefig(output_3d_png, dpi=300)
    fig3d.savefig(output_3d_svg, dpi=300)
    print(f"Saved 3D figure: {output_3d_png}")

    # ==========================================
    # FIGURE 2: 2D Radial Profiles
    # ==========================================
    fig2d = plt.figure(figsize=(7, 5.5))
    ax2d = fig2d.add_subplot(111)

    for prof in profiles:
        ax2d.plot(prof["r"], prof["z"], color=prof["color"], lw=2.5, label=prof["label"])

    ax2d.set_xlabel("Radius r (nm)", fontsize=17)
    ax2d.set_ylabel("dE (meV)", fontsize=17)
    ax2d.set_title("Radial energy profiles", fontsize=FS_TITLE, pad=12)

    ax2d.tick_params(axis="both", which="major", labelsize=FS_TICK)
    ax2d.set_xlim(0, 60)

    # Auto y-limits from all profiles
    all_z = np.concatenate([prof["z"] for prof in profiles])
    ax2d.set_ylim(np.min(all_z) - 1.0, np.max(all_z) * 1.05)

    ax2d.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
    ax2d.legend(fontsize=FS_LEGEND, loc="best")

    plt.tight_layout()
    fig2d.savefig(output_2d_png, dpi=300)
    fig2d.savefig(output_2d_svg, dpi=300)
    print(f"Saved 2D figure: {output_2d_png}")

    plt.show()


if __name__ == "__main__":
    main()
