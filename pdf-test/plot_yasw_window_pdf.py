# -*- coding: utf-8 -*-
"""
从 yasw.txt 截取指定起始时间之后一个窗长的数据，做去趋势、标准化得 Z，
拟合得到 λ，绘制该段数据的经验 PDF（红圈）与理论 PDF（黑线）。
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# 使可导入 pdf_anomaly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pdf-related-anomaly"))
from pdf_anomaly import load_data, filt_db, pdf_act, pdf_fit, _pdf_fun_vectorized

# ---------- 可改参数 ----------
YASW_FILE = Path(__file__).resolve().parent.parent / "pdf-related-anomaly" / "yasw.txt"
START_TIME = 200804100000  # 起始时间 yyyymmddHHMM，即 2008-04-20 00:00
WINDOW_LEN = 30 * 24 * 60   # 一个窗长（点数），30 天分钟
WAVELET = "db5"
LEVEL = 4
BIN_WIDTH = 0.1
LNSS_GRID = np.linspace(-100, 100, 2001)


def time_12_to_yyyymmdd(t: float) -> str:
    """12 位时间码 -> YYYYMMDD 字符串。"""
    t = int(t)
    if t >= 1e10:
        y = t // 10**8
        r = t % 10**8
        m = r // 10**6
        d = (r % 10**6) // 10**4
        return f"{y:04d}{m:02d}{d:02d}"
    return str(t)[:8]


def main():
    data = load_data(YASW_FILE)
    if len(data) == 0:
        raise FileNotFoundError(f"无数据或文件不存在: {YASW_FILE}")
    time_col = data[:, 0]
    value_col = data[:, 1]

    idx = np.where(time_col >= START_TIME)[0]
    if len(idx) == 0:
        raise ValueError(f"未找到时间 >= {START_TIME} 的数据")
    start = int(idx[0])
    end = start + WINDOW_LEN
    if end > len(time_col):
        raise ValueError(f"从 {START_TIME} 起不足 {WINDOW_LEN} 个点，请换起始时间或缩短窗长")

    detrend = filt_db(value_col, WAVELET, LEVEL)
    seg = detrend[start:end]
    seg_std = np.std(seg)
    if seg_std <= 0:
        seg_std = 1.0
    Z = seg / seg_std  # 标准化后 Z 的 std=1

    # 拟合与理论 PDF 中的 sgm 必须用“标准化序列”的标准差（=1），与主程序一致，否则 λ 和曲线都会错
    sgm_for_fit = 1.0
    in_x, in_y = pdf_act(Z, x_min=-4, x_max=4, step=BIN_WIDTH)
    lmd, _ = pdf_fit(in_x, in_y, sgm_for_fit, bin_width=BIN_WIDTH)

    # 理论 PDF 密度（不乘 bin_width），sgm=1 对应标准化变量 z
    f = _pdf_fun_vectorized(lmd, in_x, sgm_for_fit, LNSS_GRID)
    theory_density = np.trapz(f, LNSS_GRID, axis=0)
    empirical_density = in_y / BIN_WIDTH

    t_start_str = time_12_to_yyyymmdd(time_col[start])
    t_end_str = time_12_to_yyyymmdd(time_col[end - 1])
    xlabel_str = r"$Z/\sigma$ (" + t_start_str + "-" + t_end_str + ")"

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(in_x, empirical_density, "ro", markersize=4, label="Empirical PDF")
    ax.plot(in_x, theory_density, "k-", linewidth=1.2, label="Theoretical PDF")
    ax.set_ylabel(r"$P(Z)$")
    ax.set_xlabel(xlabel_str)
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-5)
    ax.set_xlim(-4, 4)
    ax.text(0.05, 0.95, r"$\lambda=%.3f$" % lmd, transform=ax.transAxes, fontsize=11, verticalalignment="top", horizontalalignment="left")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    out_dir = Path(__file__).resolve().parent
    out_file = out_dir / "yasw_window_pdf.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"图件已保存: {out_file}")


if __name__ == "__main__":
    main()
