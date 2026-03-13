# -*- coding: utf-8 -*-
"""
生成 3 组 30 天分钟采样数据，并绘制 6 个子图：上排为 3 组时序，下排为对应的 PDF 分布。

- 第 1 组：均值为 0、标准差为 1 的正态分布信号
- 第 2 组：对数正态信号（ln(X) ~ N(0, 0.6²)，即尺度参数 0.6）
- 第 3 组：第 1 组与第 2 组的逐点乘积
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 固定随机种子便于复现
np.random.seed(42)

# 30 天分钟采样点数
N = 30 * 24 * 60  # 43200

# 第 1 组：N(0, 1)
data1 = np.random.normal(0, 1, N)

# 第 2 组：对数正态，令 Z~N(0, 0.6)，X = exp(Z)，即 ln(X) 均值为 0、标准差为 0.6
data2 = np.exp(np.random.normal(0, 0.6, N))

# 第 3 组：逐点乘积
data3 = data1 * data2

# 时间轴（分钟索引，可视为从 0 开始的分钟）
t = np.arange(N, dtype=float)

# 6 个子图：上排 3 个时序，下排 3 个 PDF
fig, axes = plt.subplots(2, 3, figsize=(12, 7))

# 上排：时序（无 title）
axes[0, 0].plot(t, data1, "b", linewidth=0.4, alpha=0.8)
axes[0, 0].set_ylabel(r"$X\sim N(0, 1.0)$")
axes[0, 0].set_xlabel("Time (min)")

axes[0, 1].plot(t, data2, "g", linewidth=0.4, alpha=0.8)
axes[0, 1].set_ylabel(r"$Y\sim \ln\mathcal{N}(0, 0.6)$")
axes[0, 1].set_xlabel("Time (min)")

axes[0, 2].plot(t, data3, "r", linewidth=0.4, alpha=0.8)
axes[0, 2].set_ylabel(r"$Z=XY$")
axes[0, 2].set_xlabel("Time (min)")

# 下排：PDF（归一化直方图，y 轴对数坐标，x 轴 -4~4；第 1、2 组叠加理论曲线）
sigma_val = 1.0
lambda_val = 0.6

# 固定 x 轴 [-4,4] 的等宽 bin 边，避免 λ 增大时数据范围变大使 bin 变宽
BIN_EDGES = np.linspace(-4, 4, 161)

def plot_pdf(ax, data, color, ylabel_str, xlabel_str, theoretical_x=None, theoretical_pdf=None):
    ax.hist(data, bins=BIN_EDGES, density=True, color=color, alpha=0.7, edgecolor="none", label="Empirical")
    if theoretical_x is not None and theoretical_pdf is not None:
        ax.plot(theoretical_x, theoretical_pdf, "k--", linewidth=1.2, label="Theoretical")
    ax.set_ylabel(ylabel_str)
    ax.set_xlabel(xlabel_str)
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-5)
    ax.set_xlim(-4, 4)
    ax.legend(loc="upper right", fontsize=8)

# Group 1 理论 PDF: N(0,1)，横轴为 X/σ（σ=1）
x1 = np.linspace(-4, 4, 200)
pdf1_theory = np.exp(-0.5 * x1**2) / np.sqrt(2 * np.pi)
plot_pdf(axes[1, 0], data1, "blue", r"$P(X)$", r"$X/\sigma$", x1, pdf1_theory)
axes[1, 0].text(0.05, 0.95, r"$\sigma=1.0$", transform=axes[1, 0].transAxes, fontsize=10, verticalalignment="top", horizontalalignment="left")

# Group 2：横轴为 Y/λ（对数正态 Y 仅取正值，左侧为空正确）
s = lambda_val
x2 = np.linspace(1e-3, 4, 200)
x2 = np.maximum(x2, 1e-12)
y_vals = lambda_val * x2
pdf2_y = np.exp(-0.5 * (np.log(y_vals) / s) ** 2) / (y_vals * s * np.sqrt(2 * np.pi))
pdf2_theory = lambda_val * pdf2_y  # PDF of U=Y/λ
plot_pdf(axes[1, 1], data2 / s, "green", r"$P(Y)$", r"$Y/\lambda$", x2, pdf2_theory)
axes[1, 1].text(0.05, 0.95, r"$\lambda=0.6$", transform=axes[1, 1].transAxes, fontsize=10, verticalalignment="top", horizontalalignment="left")

# Group 3 仅经验 PDF，横轴为 Z/std(Z)
z_std = np.std(data3)
if z_std <= 0:
    z_std = 1.0
data3_norm = data3 / z_std
plot_pdf(axes[1, 2], data3_norm, "red", r"$P(Z)$", r"$Z/\mathrm{std}(Z)$")
axes[1, 2].text(0.05, 0.95, r"$\sigma=1.0$, $\lambda=0.6$", transform=axes[1, 2].transAxes, fontsize=10, verticalalignment="top", horizontalalignment="left")

# 下排 y 轴范围统一
y_min, y_max = 1e-5, 0
for ax in axes[1, :]:
    ax.set_xlim(-4, 4)
    ymn, ymx = ax.get_ylim()
    y_min = min(y_min, ymn)
    y_max = max(y_max, ymx)
for ax in axes[1, :]:
    ax.set_ylim(y_min, y_max)

plt.tight_layout()
out_dir = Path(__file__).resolve().parent
out_file = out_dir / "three_groups_and_pdf.png"
plt.savefig(out_file, dpi=150)
plt.close()
print(f"图件已保存: {out_file}")

# ========== 第 2 张图：3 行，每行仅 PDF（无观测时序），σ=1 固定，λ 分别为 0.8 / 1.3 / 1.8 ==========
def draw_pdf_row(axes_row, sigma, lam, n_sample=43200, seed=None):
    """在一行 3 个子图上绘制 P(X)、P(Y)、P(Z)，横轴 -4~4，y 对数。"""
    if seed is not None:
        np.random.seed(seed)
    X = np.random.normal(0, sigma, n_sample)
    Y = np.exp(np.random.normal(0, lam, n_sample))
    Z = X * Y
    z_std = np.std(Z)
    if z_std <= 0:
        z_std = 1.0
    Z_norm = Z / z_std

    # 左：P(X)，X/σ
    x1 = np.linspace(-4, 4, 200)
    pdf1 = np.exp(-0.5 * (x1**2)) / np.sqrt(2 * np.pi)
    axes_row[0].hist(X / sigma, bins=BIN_EDGES, density=True, color="blue", alpha=0.7, edgecolor="none", label="Empirical")
    axes_row[0].plot(x1, pdf1, "k--", linewidth=1.2, label="Theoretical")
    axes_row[0].set_ylabel(r"$P(X)$")
    axes_row[0].set_xlabel(r"$X/\sigma$")
    axes_row[0].set_yscale("log")
    axes_row[0].set_ylim(bottom=1e-5)
    axes_row[0].set_xlim(-4, 4)
    axes_row[0].legend(loc="upper right", fontsize=8)
    axes_row[0].text(0.05, 0.95, r"$\sigma=1.0$", transform=axes_row[0].transAxes, fontsize=10, verticalalignment="top", horizontalalignment="left")

    # 中：P(Y)，Y/λ（左侧为空）
    x2 = np.linspace(1e-3, 4, 200)
    x2 = np.maximum(x2, 1e-12)
    y_vals = lam * x2
    pdf2_y = np.exp(-0.5 * (np.log(y_vals) / lam) ** 2) / (y_vals * lam * np.sqrt(2 * np.pi))
    pdf2_u = lam * pdf2_y
    axes_row[1].hist(Y / lam, bins=BIN_EDGES, density=True, color="green", alpha=0.7, edgecolor="none", label="Empirical")
    axes_row[1].plot(x2, pdf2_u, "k--", linewidth=1.2, label="Theoretical")
    axes_row[1].set_ylabel(r"$P(Y)$")
    axes_row[1].set_xlabel(r"$Y/\lambda$")
    axes_row[1].set_yscale("log")
    axes_row[1].set_ylim(bottom=1e-5)
    axes_row[1].set_xlim(-4, 4)
    axes_row[1].legend(loc="upper right", fontsize=8)
    axes_row[1].text(0.05, 0.95, r"$\lambda=%s$" % lam, transform=axes_row[1].transAxes, fontsize=10, verticalalignment="top", horizontalalignment="left")

    # 右：P(Z)，Z/std(Z)
    axes_row[2].hist(Z_norm, bins=BIN_EDGES, density=True, color="red", alpha=0.7, edgecolor="none", label="Empirical")
    axes_row[2].set_ylabel(r"$P(Z)$")
    axes_row[2].set_xlabel(r"$Z/\mathrm{std}(Z)$")
    axes_row[2].set_yscale("log")
    axes_row[2].set_ylim(bottom=1e-5)
    axes_row[2].set_xlim(-4, 4)
    axes_row[2].legend(loc="upper right", fontsize=8)
    axes_row[2].text(0.05, 0.95, r"$\sigma=1.0$, $\lambda=%s$" % lam, transform=axes_row[2].transAxes, fontsize=10, verticalalignment="top", horizontalalignment="left")

fig2, axes2 = plt.subplots(3, 3, figsize=(12, 8))
params = [(1.0, 0.8), (1.0, 1.3), (1.0, 1.8)]
for i, (sig, lam) in enumerate(params):
    draw_pdf_row(axes2[i, :], sig, lam, seed=42 + i)
# 每行 y 轴范围统一
for i in range(3):
    y_min, y_max = 1e-5, 0
    for ax in axes2[i, :]:
        ax.set_xlim(-4, 4)
        ymn, ymx = ax.get_ylim()
        y_min = min(y_min, ymn)
        y_max = max(y_max, ymx)
    for ax in axes2[i, :]:
        ax.set_ylim(y_min, y_max)
plt.tight_layout()
out_file2 = out_dir / "pdf_three_lambdas.png"
plt.savefig(out_file2, dpi=150)
plt.close()
print(f"图件已保存: {out_file2}")

# 可选：将 3 组数据写出为文本，便于 pdf_anomaly 等程序使用
# 若需要可取消下面注释，并指定时间列为 yyyymmddHHMM 格式
# t_base = 202001010000  # 示例：2020-01-01 00:00 起每分钟
# with open(out_dir / "group1.txt", "w") as f:
#     for i in range(N):
#         f.write(f"{t_base + i}\t{data1[i]:.4f}\n")
