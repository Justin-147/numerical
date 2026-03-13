# -*- coding: utf-8 -*-
"""
基于概率密度拟合的异常分析（PDF-related anomaly）。
小波去趋势 + 滑动窗口拟合 λ²，输出时序 λ² 及误差。

程序入口：运行本文件，在 __main__ 中配置数据路径与参数后执行。
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, message=".*PyWavelets.*")

# ===========================
# 一、数据读取与预处理
# ===========================


def load_data(filepath: Path, missing_values: Tuple[float, ...] = (99999.0, 999999.0)) -> np.ndarray:
    """
    加载两列数据：第 1 列时间码（yyyymmddHHMM），第 2 列分钟观测值。
    仅支持分钟采样数据。
    对 99999、999999 等缺测：直接剔除对应行，不做插值。
    返回 (N, 2) 的数组：col0 时间，col1 数值。
    """
    data = np.loadtxt(filepath, dtype=float, ndmin=2)
    if data.size == 0:
        return data
    mask = np.ones(len(data), dtype=bool)
    for mv in missing_values:
        mask &= (data[:, 1] != mv)
    return data[mask]


# ===========================
# 二、小波去趋势（filt_db）
# ===========================


def filt_db(y: np.ndarray, wavelet: str, level: int) -> np.ndarray:
    """
    小波分解后仅重建指定层级的细节分量，作为“去趋势”后的序列（高频信息）。
    """
    import pywt
    y = np.asarray(y, dtype=float).ravel()
    coeffs = pywt.wavedec(y, wavelet, level=level)
    # coeffs = [cA_n, cD_n, cD_{n-1}, ..., cD_1]，细节层 level 对应 coeffs[1]
    detail_coeffs = [np.zeros_like(c) for c in coeffs]
    detail_coeffs[1] = coeffs[1]
    rec = pywt.waverec(detail_coeffs, wavelet)
    rec = np.asarray(rec, dtype=float)
    if len(rec) > len(y):
        rec = rec[: len(y)]
    elif len(rec) < len(y):
        rec = np.resize(rec, len(y))
    return rec


# ===========================
# 三、经验 PDF + 拟合 λ（积分 over lns）
# ===========================


def pdf_act(z: np.ndarray, x_min: float = -4.0, x_max: float = 4.0, step: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    经验 PDF：将标准化序列 z 在 [x_min, x_max] 上按步长 step 做直方图，得到各 bin 中心与概率（频数/总数）。
    中心为各 bin 中点（如 -3.95, -3.85），按小数位舍入以消除浮点误差。
    """
    edges = np.arange(x_min, x_max + step * 0.5, step)
    count, _ = np.histogram(z, bins=edges)
    prob = count / max(z.size, 1)
    centers = (edges[:-1] + edges[1:]) / 2
    decimals = max(1, round(-np.log10(step)) + 1)
    centers = np.round(centers, decimals)
    return centers, prob


def _pdf_fun(lmd: float, z: float, sgm: float, lns: float) -> float:
    """理论 PDF 被积函数：dlt=exp(lns), fs=N(z/sgm;0,dlt^2), gs=N(lns;0,lmd^2), mf=gs*fs/dlt."""
    dlt = np.exp(lns)
    if dlt <= 0:
        return 0.0
    fsx = z / sgm
    fs = (1.0 / (sgm * np.sqrt(2 * np.pi))) * np.exp(-(fsx**2) / (2 * dlt**2))
    gs = (1.0 / (lmd * np.sqrt(2 * np.pi))) * np.exp(-(lns**2) / (2 * lmd**2))
    return float(gs * fs / dlt)


def _pdf_fun_vectorized(
    lmd: float,
    z: np.ndarray,
    sgm: float,
    lns: np.ndarray,
) -> np.ndarray:
    """向量化：lns (L,), z (Z,) -> 输出 (L, Z)，便于沿 lns 积分得每个 z 的 PDF。"""
    dlt = np.exp(lns)
    dlt = np.maximum(dlt, 1e-300)
    # lns (L,), z (Z,) -> fsx (1,Z), dlt (L,1) -> fs (L,Z)
    fsx = np.asarray(z, dtype=float).reshape(1, -1) / sgm
    dlt2 = (dlt.reshape(-1, 1) ** 2)
    fs = (1.0 / (sgm * np.sqrt(2 * np.pi))) * np.exp(-(fsx**2) / (2 * dlt2))
    gs = (1.0 / (lmd * np.sqrt(2 * np.pi))) * np.exp(-(lns**2) / (2 * lmd**2))
    gs = gs.reshape(-1, 1)
    dlt = dlt.reshape(-1, 1)
    return gs * fs / dlt


def pdf_fit(
    in_x: np.ndarray,
    in_y: np.ndarray,
    sgm: float,
    bin_width: float = 0.1,
) -> Tuple[float, float]:
    """
    这个函数用于 PDF 拟合：给定经验 PDF (in_x, in_y) 和标准差 sgm，在指定范围和步长内“搜索”最优的 lmd，使理论 PDF 最接近经验 PDF。
    返回 (best_lmd, best_err)。

    主要思路是：对 lmd 取不同的值，算出对应理论 PDF，然后和经验 PDF 做残差评估，找出最优的 lmd 和误差。
    这里通过 ln(sigma) 网格积分+向量化，大幅提升速度。
    """

    # lns_grid 对应 ln(sigma) 的取值区间[-100, 100]，共2001个点，积分用。
    lns_grid = np.linspace(-100, 100, 2001)

    # kstps 是三轮粗到精的步长（1.0, 0.1, 0.01）
    kstps = [1.0, 0.1, 0.01]
    tmp_lmd = 0.01  # 记录上一轮搜到的最优lmd，为下一轮搜索区间的中心
    best_err = 0.0  # 存储最优lmd对应的误差

    # 三轮从粗到细的依据与参数设置说明：
    # - 第一轮用 [0.01, 10.01) 和步长 1.0，为了快速覆盖理论上 λ 可能的所有主流范围（大部分实际分布不会超过这个区间），
    #   避免最优解落在区间外而漏检。步长大是为了高效、初步锁定大致值。
    # - 第二轮和第三轮则均以上一轮最优 λ 为中心（tmp_lmd），两侧各 10 步，窗口宽度分别为 2.0（0.1x20）、0.2（0.01x20），
    #   步长递减（0.1 → 0.01），确保不会因步长过大错过最优点，也不会范围过窄遗漏极值——这样能够兼顾“全局-局部”、
    #   扫描“全面性”与“聚焦性”，且通过 max(0.01, ...) 及 min(..., 100.02) 控制 λ 合理取值范围，防止越界。
    # - 这种做法会让搜索空间逐步收敛，最大程度减少遗漏，同时专注越来越精的 λ 最优区间，聚焦最优值。
    for k in range(3):
        if k == 0:
            # 第1轮：大步长粗覆盖，防止全局最优被忽略
            lmds = np.arange(0.01, 10.01, kstps[k])
        else:
            # 第2、3轮：以上一轮最优 λ 为中心，两边各 10 步、步长递减进行细化搜索，保证精度与收敛
            lo = max(0.01, tmp_lmd - kstps[k] * 10)
            hi = tmp_lmd + kstps[k] * 10
            lmds = np.arange(lo, min(hi + kstps[k] * 0.5, 100.02), kstps[k])

        if len(lmds) == 0:
            continue

        # 下面这段对每个lmd值都算一次理论pdf（对 in_x 里的每个点算积分），整体向量化，加速。
        ps = np.zeros((len(in_x), len(lmds)))  # 存放各lmd下理论pdf
        for j, lmd in enumerate(lmds):
            f = _pdf_fun_vectorized(lmd, in_x, sgm, lns_grid)
            # 对于每个lmd，在lns_grid上对 f 积分（axis=0），得到理论pdf
            ps[:, j] = np.trapz(f, lns_grid, axis=0) * bin_width

        # 残差加权评估
        var_ps = np.var(ps, axis=0)
        var_ps[var_ps <= 0] = 1e-12  # 防止分母为零
        X2 = np.sum((in_y[:, None] - ps) ** 2 / (sgm**2 + var_ps), axis=0)
        errs = np.sqrt(np.sum((in_y[:, None] - ps) ** 2, axis=0))
        m = int(np.argmin(X2))
        tmp_lmd = float(lmds[m])   # 找到这一轮最优的lmd
        best_err = float(errs[m])  # 最优的残差

    # 返回最终最优的lmd和误差
    return tmp_lmd, best_err


# ===========================
# 四、滑动窗口主流程
# ===========================


def run_pdf_anomaly(
    data_file: Path,
    output_dir: Path | None = None,
    wavelet: str = "db5",
    level: int = 4,
    window_minutes: int | None = None,
    step_minutes: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    主流程：加载数据 → 去趋势 → 滑动窗口拟合求 λ² 与误差 → 返回并写文件、绘图。

    缺测与窗长：缺测（99999/999999）直接剔除、不插值。窗长/步长为“点数”（如 30×24×60 个采样点），
    非日历时长；剔除缺测后，相同点数对应的实际时间跨度可能不等，各窗的日历范围会不一致。

    默认 30 天窗长、1 天步长，经验 PDF -4~4/0.1。返回 (time_end, lmd2, err, detrend_series)。
    """
    print("正在加载数据...")
    data = load_data(data_file)
    if len(data) == 0:
        raise ValueError("无有效数据")
    time_col = data[:, 0]
    value_col = data[:, 1]
    print(f"  有效数据点数: {len(data)}")

    print("正在小波去趋势...")
    detrend = filt_db(value_col, wavelet, level)

    wl = window_minutes if window_minutes is not None else 30 * 24 * 60
    stp = step_minutes if step_minutes is not None else 24 * 60

    n = len(value_col)
    n_windows = len(range(0, n - wl, stp))
    print(f"滑动窗口拟合 (窗长={wl} 点, 步长={stp} 点, 共 {n_windows} 个窗口)...")
    td_list = []
    lmd2_list = []
    err_list = []

    for iw, start in enumerate(range(0, n - wl, stp)):
        end = start + wl
        de_y = detrend[start:end]
        if np.std(de_y) <= 0:
            continue
        zds = de_y / np.std(de_y)
        t_end = time_col[end - 1]
        pdf_x, pdf_y = pdf_act(zds, x_min=-4, x_max=4, step=0.1)
        sgm = np.std(zds)
        if sgm <= 0:
            sgm = 1.0
        lmd, err = pdf_fit(pdf_x, pdf_y, sgm)
        td_list.append(t_end)
        lmd2_list.append(lmd**2)
        err_list.append(err)
        if (iw + 1) % max(1, n_windows // 10) == 0 or (iw + 1) == n_windows:
            print(f"  已处理 {iw + 1}/{n_windows} 个窗口")

    time_end = np.array(td_list)
    lmd2 = np.array(lmd2_list)
    errs = np.array(err_list)

    out_dir = output_dir or data_file.parent
    stem = data_file.stem
    # 去趋势序列输出：例如yasw_4_db5_detrend.txt
    detrend_file = out_dir / f"{stem}_{level}_{wavelet}_detrend.txt"
    np.savetxt(detrend_file, np.column_stack([time_col, detrend]), fmt="%.0f\t%.4f", delimiter="\t")
    print(f"已保存去趋势文件: {detrend_file}")
    # λ² 与误差输出：例如yasw_4_db5_lmd.txt
    lmd_file = out_dir / f"{stem}_{level}_{wavelet}_lmd.txt"
    np.savetxt(lmd_file, np.column_stack([time_end, lmd2, errs]), fmt="%.0f\t%.4f\t%.6f", delimiter="\t")
    print(f"已保存 lmd 结果文件: {lmd_file}")

    # 绘图并保存
    try:
        _plot_result(data_file, out_dir, stem, time_col, value_col, detrend, time_end, lmd2, errs, level, wavelet)
    except Exception as e:
        import traceback
        print(f"绘图失败: {e}")
        traceback.print_exc()

    return time_end, lmd2, errs, detrend


def _plot_result(
    data_file: Path,
    out_dir: Path,
    stem: str,
    time_col: np.ndarray,
    value_col: np.ndarray,
    detrend: np.ndarray,
    time_end: np.ndarray,
    lmd2: np.ndarray,
    errs: np.ndarray,
    level: int,
    wavelet: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")  # 无界面后端，确保可保存图件
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    # 图件中可能含中文（如文件名 stem），设置支持中文的字体，避免 Glyph missing 警告
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun", "KaiTi", "FangSong", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示

    def _time_to_datetime(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        # 支持 8 位 yyyymmdd 或 12 位 yyyymmddHHMM
        s = np.zeros(len(t), dtype="datetime64[s]")
        for i, v in enumerate(t):
            v = int(v)
            if v >= 1e10:
                # 12 位: yyyymmddHHMM -> 年4位、月2位、日2位、时2位、分2位
                y = v // 10**8
                r = v % 10**8
                m = r // 10**6
                r = r % 10**6
                d = r // 10**4
                r = r % 10**4
                h = r // 100
                mi = r % 100
                s[i] = np.datetime64(f"{int(y):04d}-{int(m):02d}-{int(d):02d}T{int(h):02d}:{int(mi):02d}:00")
            else:
                # 8 位: yyyymmdd -> 日界 00:00:00，与 12 位统一为带时间的 datetime64[s] 便于绘图
                y = v // 10000
                r = v % 10000
                m = r // 100
                d = r % 100
                s[i] = np.datetime64(f"{int(y):04d}-{int(m):02d}-{int(d):02d}T00:00:00")
        return s

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    dt_raw = _time_to_datetime(time_col)
    dt_lmd = _time_to_datetime(time_end)

    x_min = min(dt_raw.min(), dt_lmd.min())
    x_max = max(dt_raw.max(), dt_lmd.max())
    for ax in axes:
        ax.set_xlim(x_min, x_max)

    axes[0].plot(dt_raw, value_col, "k", linewidth=0.6)
    axes[0].set_ylabel("Raw value")

    axes[1].plot(dt_raw, detrend, "b", linewidth=0.6)
    axes[1].set_ylabel("Detrended (high-freq)")

    axes[2].errorbar(dt_lmd, lmd2, yerr=errs, fmt="r.", markersize=4, ecolor="blue", capsize=2)
    axes[2].set_ylabel(r"$\lambda^2$")
    axes[2].set_xlabel("Time")

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.suptitle(f"{stem}  level={level} {wavelet}")
    plt.tight_layout()
    out_png = out_dir / f"{stem}_{level}_{wavelet}.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    if out_png.exists():
        print(f"图件已保存: {out_png}")
    else:
        print(f"警告: 图件保存可能失败，请检查路径 {out_png}")


# ===========================
# 五、程序入口
# ===========================


if __name__ == "__main__":
    from pathlib import Path

    # ---------- 可修改参数 ----------
    DATA_FILE = Path(r"d:\numerical\pdf-related-anomaly\yasw.txt")
    OUTPUT_DIR = None  # 默认与数据文件同目录
    WAVELET = "db5" # 小波基函数
    LEVEL = 4 # 小波分解层数
    WINDOW_MINUTES = 30 * 24 * 60 # 窗长30天
    STEP_MINUTES = 24 * 60 # 步长1天

    if not DATA_FILE.exists():
        print("请先准备分钟采样数据文件（两列：时间码、观测值），或修改 DATA_FILE 路径。")
    else:
        run_pdf_anomaly(
            data_file=DATA_FILE,
            output_dir=OUTPUT_DIR,
            wavelet=WAVELET,
            level=LEVEL,
            window_minutes=WINDOW_MINUTES,
            step_minutes=STEP_MINUTES,
        )
        print("PDF 异常分析完成。")
