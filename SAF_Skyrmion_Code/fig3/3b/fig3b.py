import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib as mpl

# -------- 1. 修改配置：调大字体 --------
mpl.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.labelsize": 16,    # 坐标轴标签字号 (加大)
    "xtick.labelsize": 13,   # X轴刻度字号 (加大)
    "ytick.labelsize": 13,   # Y轴刻度字号 (加大)
    "axes.linewidth": 1.0,   # 边框稍微加粗
    "figure.dpi": 150,
})

# 参数设置
eta   = 1.23
beta1 = 0.0316
beta2 = 0.0510

def Rc(l1, l2, eta, beta1, beta2):
    D1 = eta**2 * (beta1 + beta2*l1)**2 + (1.0 - l1)**2
    D2 = eta**2 * (beta1 + beta2*l2)**2 + (1.0 - l2)**2
    return np.sqrt(D1 / D2)

# 网格生成
n = 301
lam = np.linspace(0.0, 1.0, n)
L1, L2 = np.meshgrid(lam, lam)
Rc_grid = Rc(L1, L2, eta, beta1, beta2)

vmin = Rc_grid.min()
vmax = min(10.0, Rc_grid.max())
norm = colors.PowerNorm(gamma=0.4, vmin=vmin, vmax=vmax)

# 绘图 (稍微调大图片尺寸以容纳大字体)
fig, ax = plt.subplots(figsize=(4.2, 3.8))

im = ax.pcolormesh(
    L1, L2, Rc_grid,
    shading="auto",
    cmap="turbo",
    norm=norm,
)

# Colorbar 设置
cbar = fig.colorbar(
    im,
    ax=ax,
    fraction=0.055,
    pad=0.04,
    shrink=0.85,
)
# 加大 colorbar 字体
# cbar.set_label(r"$R_c(\lambda_1,\lambda_2)$", fontsize=15)
cbar.ax.tick_params(width=0.8, length=4, labelsize=12)

# 等高线绘制
levels = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0])
levels = levels[(levels >= vmin) & (levels <= vmax)]

cs = ax.contour(
    L1, L2, Rc_grid,
    levels=levels,
    colors="k",
    linewidths=0.6, # 稍微加粗线条
    alpha=0.8
)

# 等高线标签 (加大字体)
texts = ax.clabel(
    cs,
    cs.levels[::2],
    fmt="%.1f",
    fontsize=10,  # 加大这里的字体
    inline=False, # 如果觉得挡住线了，可以改成 True
)

# 标签位置微调
x0, x1 = ax.get_xlim()
y0, y1 = ax.get_ylim()
cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
shrink = 0.8
for t in texts:
    x, y = t.get_position()
    dx, dy = x - cx, y - cy
    t.set_position((cx + shrink * dx, cy + shrink * dy))

# -------- 2. 轴与原点特殊处理 --------
ax.set_xlabel(r"$\lambda_1$")
ax.set_ylabel(r"$\lambda_2$")
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.set_aspect("equal", adjustable="box")
ax.tick_params(direction="in", top=True, right=True, width=0.8, length=4)

# 设定固定的刻度
ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
ax.set_xticks(ticks)
ax.set_yticks(ticks)

# 隐藏坐标轴自带的 '0.0' 标签
# get_xticklabels 获取的是对象列表，我们需要把第一个设为空字符串
xticklabels = [str(t) if t != 0 else "" for t in ticks]
yticklabels = [str(t) if t != 0 else "" for t in ticks]
ax.set_xticklabels(xticklabels)
ax.set_yticklabels(yticklabels)

# 手动在左下角添加一个共用的 "0"
# transform=ax.transAxes 确保坐标是相对于轴的 (0,0) 位置
# 稍微加一点负偏移 (-0.03, -0.02) 让它不要贴得太紧
ax.text(-0.035, -0.035, "0", transform=ax.transAxes, 
        ha='right', va='top', fontsize=13)

plt.tight_layout()

plt.savefig("Rc_phase2_large_font.png", format="png", bbox_inches="tight")
print("Image saved as Rc_phase2_large_font.png")

plt.show()