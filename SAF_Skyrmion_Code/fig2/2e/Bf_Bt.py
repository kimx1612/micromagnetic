#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
绘图脚本：
- 长宽比优化：更宽的矩形 (Wider Aspect Ratio)
- 输出 SVG 矢量图
- 字号极大 & 刻度稀疏 (Nature Style)
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ============================
# 1. 基础设置
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCAN_ROOT = os.path.join(BASE_DIR, "Data_Scan")

DEMO_MODE = not os.path.isdir(SCAN_ROOT)

re_current = re.compile(r"^#\s*Current_J\s*=\s*([0-9eE+\-\.]+)")
re_xi      = re.compile(r"^#\s*Xi_layer3\s*=\s*([0-9eE+\-\.]+)")
re_beta1   = re.compile(r"^#\s*beta1\s*=\s*([0-9eE+\-\.]+)")

def parse_fit_file(path):
    J_val, xi_val, beta1 = None, None, None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if m := re_current.match(line): J_val = float(m.group(1))
            elif m := re_xi.match(line): xi_val = float(m.group(1))
            elif m := re_beta1.match(line): beta1 = float(m.group(1))
    return J_val, xi_val, beta1

def parse_J(name):
    try: return abs(float(name.split("J_")[1]))
    except: return 0.0

def sci_tex(x):
    """LaTeX 格式化"""
    if x == 0: return r"$0$"
    x = abs(x)
    s = f"{x:.0e}"
    m, e = s.split("e")
    if float(m) == 1.0:
        return rf"$10^{{{int(e)}}}$"
    return rf"${float(m):g}\times10^{{{int(e)}}}$"

# ============================
# 2. 数据处理
# ============================
groups = []
if not DEMO_MODE:
    J_folders = sorted([d for d in os.listdir(SCAN_ROOT) if d.startswith("J_")], 
                       key=parse_J)
    for Jf in J_folders:
        path = os.path.join(SCAN_ROOT, Jf)
        entries = []
        for xid in os.listdir(path):
            fp = os.path.join(path, xid, "fit_result.txt")
            if os.path.isfile(fp):
                try:
                    res = parse_fit_file(fp)
                    if None not in res: entries.append(res)
                except: pass
        if entries:
            entries.sort(key=lambda x: x[1])
            groups.append({
                "J": parse_J(Jf),
                "xi": np.array([e[1] for e in entries]),
                "beta": np.array([e[2] for e in entries])
            })
else:
    # 演示数据
    print("生成演示数据...")
    for J in [1e11, 3e11, 5e11]:
        xi = np.linspace(0.02, 0.15, 20)
        beta = xi * (1.0 + np.random.normal(0, 0.04, xi.shape))
        groups.append({"J": J, "xi": xi, "beta": beta})

plot_data = []
NUM_PER_J = 4
STRIDE = max(1, len(groups))

for gid, g in enumerate(groups):
    indices = list(range(gid, len(g["xi"]), STRIDE))[:NUM_PER_J]
    for idx in indices:
        plot_data.append((g["J"], g["xi"][idx], g["beta"][idx], gid))

# ============================
# 3. 绘图 (矩形、SVG)
# ============================
if plot_data:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 14,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 13,
        "legend.title_fontsize": 14,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 300,
    })

    colors = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F"]
    markers = ["o", "s", "^", "D", "v"]

    # === 关键修改：调整长宽比 ===
    # 5.2 (宽) x 3.6 (高) -> 看起来更舒展，不像正方形那么局促
    fig, ax = plt.subplots(figsize=(5.2, 3.6))

    # 绘制参考线
    all_vals = [p[1] for p in plot_data]
    if all_vals:
        mn, mx = min(all_vals), max(all_vals)
        pad = (mx - mn) * 0.1
        ax.plot([mn-pad, mx+pad], [mn-pad, mx+pad], 
                "--", color="gray", alpha=0.5, lw=1.5, zorder=0)

    # 绘制散点
    unique_gids = sorted(list(set(p[3] for p in plot_data)))
    for gid in unique_gids:
        sub = [p for p in plot_data if p[3] == gid]
        xs = [p[1] for p in sub]
        ys = [p[2] for p in sub]
        J_val = sub[0][0]
        
        c = colors[gid % len(colors)]
        m = markers[gid % len(markers)]
        
        ax.plot(xs, ys, ls="None",
                marker=m, markersize=8.0,
                markeredgecolor=c, markeredgewidth=1.5,
                markerfacecolor="none",
                label=sci_tex(J_val), zorder=10)

    ax.set_xlabel(r"Input $\beta_{\mathrm{true}}$")
    ax.set_ylabel(r"Fitted $\beta^{\mathrm{fit}}_{1}$")

    # 稀疏刻度
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5)) # 稍微放宽一点点，因为X轴变长了
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    # 图例
    leg = ax.legend(title=r"$J\ (\mathrm{A/m^2})$", 
                    frameon=False, loc="best",
                    handletextpad=0.1, labelspacing=0.4)
    leg._legend_box.align = "left"

    plt.tight_layout()
    
    # === 保存 SVG 和 PNG ===
    out_svg = os.path.join(BASE_DIR, "Beta_Linearity_Wide.svg")
    out_png = os.path.join(BASE_DIR, "Beta_Linearity_Wide.png")
    
    plt.savefig(out_svg, bbox_inches="tight", format="svg") # 矢量图
    plt.savefig(out_png, bbox_inches="tight", dpi=300)      # 预览图
    
    print(f"SVG矢量图已保存: {out_svg}")
    print(f"PNG预览图已保存: {out_png}")
    plt.show()
else:
    print("无数据。")