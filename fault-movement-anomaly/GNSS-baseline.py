#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GNSS-baseline.py
================

GNSS基线长度与方位角的计算与成图流程。

主要功能
--------
- 读取 CENC GNSS 单站时序（`.neu` 文本，参考位置 + N/E/U/mm）。
- 对两站数据按十进制年对齐，仅使用共同时间段。
- 采用椭球算法（WGS‑84）计算每个历元的
  - 大地线长度 S（m）
  - 正向方位角 A1（0–360°）
- 将长度和方位角：
  - 扣除首历元的值，得到相对变化量
  - 长度转为 mm，方位角转为毫秒（millisecond of degree）
  - 对两者分别做线性拟合并扣除趋势，得到去趋势序列
- 输出：
  - `<STA1>_<STA2>_baseline.txt`
  - `<STA1>_<STA2>_baseline_detrend.txt`
  - 四幅 PNG 图（原始/去趋势的 baseline 与 azimuth）

使用说明
--------
- 将 CENC 原始数据文件（如 `SCLH_raw.neu`、`SCTQ_raw.neu`）放入 `DATA_DIR` 指定目录。
- 在文件底部 `if __name__ == "__main__":` 中设置站对列表 `STATION_PAIRS`，然后运行：

  ```bash
  python GNSS-baseline.py
  ```

依赖：numpy、matplotlib（在本目录的 requirements.txt 中列出）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ===========================
# 一、CENC GNSS 单站数据读取
# ===========================


@dataclass
class GnssStationData:
    lon: float
    lat: float
    name: str
    # 列: [YYYYMMDD, YYYY.DECM, N(mm), E(mm), U(mm), sig_n, sig_e, sig_u]
    data: np.ndarray


def read_gnss_cenc(folder: Path, filename: str) -> GnssStationData:
    """
    读取中国地震台网中心（CENC）单站时序数据。

    文件格式:
    - 第一行: '#Reference position  lon  lat  height  STATION'
    - 第二行: '# YYYYMMDD YYYY.DECM N(mm) E(mm) U(mm) sig_n(mm) sig_e(mm) sig_u(mm)'
    - 之后每行: 数值
    """
    path = folder / filename
    if not path.exists():
        raise FileNotFoundError(f"GNSS 数据文件不存在: {path}")

    lon: float | None = None
    lat: float | None = None
    station: str | None = None
    rows: List[List[float]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#Reference position"):
                parts = line.split()
                # 形如: #Reference position    100.67     31.39   3257.83    SCLH
                lon = float(parts[2])
                lat = float(parts[3])
                station = parts[5]
            elif line.startswith("#"):
                continue
            else:
                parts = line.split()
                if len(parts) < 8:
                    continue
                vals = [float(x) for x in parts[:8]]
                rows.append(vals)

    if lon is None or lat is None or station is None:
        raise ValueError(f"参考位置或台站名未在文件中正确解析: {path}")

    data = np.asarray(rows, dtype=float)
    return GnssStationData(lon=lon, lat=lat, name=station, data=data)


# ===========================
# 二、基线解算
# ===========================


def compute_baseline(fai1_deg: float, lamda1_deg: float, fai2_deg: float, lamda2_deg: float) -> Tuple[float, float, float]:
    """
    计算两点间大地线长度与正/反方位角（WGS‑84）。

    参数:
        fai1_deg, lamda1_deg: 点 1 的纬度、经度（度）
        fai2_deg, lamda2_deg: 点 2 的纬度、经度（度）
    返回:
        S: 大地线长度（m）
        A1: 正向方位角（0~360 度）
        A2: 反向方位角（0~360 度）
    """
    # WGS‑84 椭球参数
    E2 = 0.00669437999013
    EP2 = 0.00673949674227
    ea = 6378137.0
    eb = 6356752.3142

    alphaTmp1 = (E2 / 2 + E2**2 / 8 + E2**3 / 16) * 1e10
    alphaTmp2 = (E2**2 / 16 + E2**3 / 16) * 1e10
    alphaTmp3 = (3 * E2**3 / 128) * 1e10
    beltaPTmp1 = 2 * (E2**2 / 32 + E2**3 / 32) * 1e10
    beltaPTmp2 = 2 * E2**3 / 64 * 1e10
    Atmp1 = eb * EP2 / 4
    Atmp2 = eb * 3 * EP2**2 / 64
    BPPtmp1 = Atmp2
    BPPtmp2 = eb * EP2**2 / 16
    CPP = eb * EP2**2 / 64

    rad = np.pi / 180.0
    sfai1 = np.sin(fai1_deg * rad)
    sfai2 = np.sin(fai2_deg * rad)
    W1 = np.sqrt(1 - E2 * sfai1**2)
    W2 = np.sqrt(1 - E2 * sfai2**2)
    su1 = sfai1 * np.sqrt(1 - E2) / W1
    su2 = sfai2 * np.sqrt(1 - E2) / W2
    cu1 = np.cos(fai1_deg * rad) / W1
    cu2 = np.cos(fai2_deg * rad) / W2
    a1 = su1 * su2
    a2 = cu1 * cu2
    b1 = cu1 * su2
    b2 = su1 * cu2
    L = lamda2_deg - lamda1_deg

    delta = 0.0
    A1 = 0.0
    for _ in range(100):
        delta0 = delta
        lamda = L + delta0
        slamda = np.sin(lamda * rad)
        clamda = np.cos(lamda * rad)
        p = cu2 * slamda
        q = b1 - b2 * clamda

        # A1=abs(atan(p/q)*180/pi);
        if q == 0.0:
            A1_tmp = 90.0 if p > 0.0 else -90.0
        else:
            A1_tmp = np.degrees(np.arctan(p / q))
        A1_tmp = abs(A1_tmp)
        if p > 0.0:
            if q < 0.0:
                A1_tmp = 180.0 - A1_tmp
        elif p < 0.0:
            if q > 0.0:
                A1_tmp = 360.0 - A1_tmp
            elif q < 0.0:
                A1_tmp = 180.0 + A1_tmp
        A1 = A1_tmp

        sA1 = np.sin(A1 * rad)
        cA1 = np.cos(A1 * rad)
        ssigma = p * sA1 + q * cA1
        csigma = a1 + a2 * clamda

        # sigma=abs(atan(ssigma/csimga));
        if csigma == 0.0:
            sigma = np.pi / 2.0
        else:
            sigma = np.abs(np.arctan(ssigma / csigma))
        if csigma < 0.0:
            sigma = np.pi - sigma

        sA0 = cu1 * sA1
        c2A0 = 1.0 - sA0**2
        x = 2.0 * a1 - c2A0 * csigma

        alpha = (alphaTmp1 - (alphaTmp2 - alphaTmp3 * c2A0) * c2A0) * 1e-10
        beltaP = (beltaPTmp1 - beltaPTmp2 * c2A0) * 1e-10
        delta = (alpha * sigma - beltaP * x * ssigma) * sA0 * 180.0 / np.pi
        if np.abs(delta * 3600.0 - delta0 * 3600.0) < 1e-12:
            break

    A = eb + (Atmp1 - Atmp2 * c2A0) * c2A0
    BPP = BPPtmp1 - BPPtmp2 * c2A0
    y = ((1.0 - sA0**2) ** 2 - 2.0 * x**2) * csigma
    S = A * sigma + (BPP * x + CPP * y) + ssigma  # m

    p2 = cu1 * slamda
    q2 = b1 * clamda - b2
    # tA2=atan(p2/q2)*180/pi;
    if q2 == 0.0:
        tA2 = 90.0 if p2 > 0.0 else -90.0
    else:
        tA2 = np.degrees(np.arctan(p2 / q2))
    A2 = abs(tA2)
    if tA2 > 0.0:
        if sA1 > 0.0:
            A2 = 180.0 + A2
    elif tA2 < 0.0:
        if sA1 > 0.0:
            A2 = 360.0 - A2
        elif sA1 < 0.0:
            A2 = 180.0 - A2

    return float(S), float(A1), float(A2)


# ===========================
# 三、单对基线处理
# ===========================


def process_gnss_to_baseline(data_dir: Path, out_dir: Path, first_file: str, second_file: str) -> None:
    """
    读取两站 CENC GNSS 数据，计算基线长度与方位角时间序列、去趋势序列并输出。
    """
    st1 = read_gnss_cenc(data_dir, first_file)
    st2 = read_gnss_cenc(data_dir, second_file)

    # 以十进制年为时间，对齐两站观测
    # 对 DECM 四舍五入到 4 位小数再求交，避免不同文件浮点表示差异导致匹配不全
    t1 = st1.data[:, 1]  # YYYY.DECM
    t2 = st2.data[:, 1]
    t1r = np.round(t1, 4)
    t2r = np.round(t2, 4)
    common_t, ind1, ind2 = np.intersect1d(t1r, t2r, return_indices=True)
    if common_t.size == 0:
        raise ValueError(f"两站 {st1.name} 与 {st2.name} 没有共同时间段。")

    use1 = st1.data[ind1, 1:4]  # [YYYY.DECM, N, E]
    use2 = st2.data[ind2, 1:4]

    ea = 6378137.0  # m
    out_data = np.zeros((use1.shape[0], 3), dtype=float)
    for i in range(use1.shape[0]):
        # N/E 为 mm；按 B1/B2 的公式转换到近似纬度/经度并叠加参考坐标
        B1 = use1[i, 1] / ea / 1000.0 + st1.lat * np.pi / 180.0
        ICB1 = np.round((np.pi / 2 - B1) * 20000.0) / 20000.0
        L1 = use1[i, 2] / (np.sin(ICB1) * ea) / 1000.0 + st1.lon * np.pi / 180.0

        B2 = use2[i, 1] / ea / 1000.0 + st2.lat * np.pi / 180.0
        ICB2 = np.round((np.pi / 2 - B2) * 20000.0) / 20000.0
        L2 = use2[i, 2] / (np.sin(ICB2) * ea) / 1000.0 + st2.lon * np.pi / 180.0

        S, A1, _ = compute_baseline(B1 * 180.0 / np.pi, L1 * 180.0 / np.pi, B2 * 180.0 / np.pi, L2 * 180.0 / np.pi)
        out_data[i, 0] = use1[i, 0]  # YYYY.DECM
        out_data[i, 1] = S
        out_data[i, 2] = A1

    # 相对变化量 + 单位转换
    outN = out_data.copy()
    outN[:, 1] = (outN[:, 1] - outN[0, 1]) * 1000.0  # baseline: m -> 相对量 mm
    outN[:, 2] = (outN[:, 2] - outN[0, 2]) * 3600.0 * 1000.0  # azimuth: 度差 -> 毫秒

    # 去趋势（线性拟合，baseline 与 azimuth 分别做一次）
    outND = outN.copy()
    t = outND[:, 0]
    p1 = np.polyfit(t, outND[:, 1], 1)
    outND[:, 1] = outND[:, 1] - np.polyval(p1, t)
    p2 = np.polyfit(t, outND[:, 2], 1)
    outND[:, 2] = outND[:, 2] - np.polyval(p2, t)

    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{st1.name}_{st2.name}"

    # 文本输出
    def _write_baseline(path: Path, arr: np.ndarray) -> None:
        with path.open("w", encoding="utf-8") as f:
            f.write("#YYYY.DECM     baseline(mm)     azimuth(millisecond)\n")
            for row in arr:
                f.write(f"{row[0]:.4f}     {row[1]:.2f}     {row[2]:.2f}\n")

    file_orig = out_dir / f"{base_name}_baseline.txt"
    file_detr = out_dir / f"{base_name}_baseline_detrend.txt"
    _write_baseline(file_orig, outN)
    _write_baseline(file_detr, outND)

    print(f"数据输出:\n  {file_orig}\n  {file_detr}")

    # 图件输出（仅 PNG）
    def _plot_series(path_png_base: Path, x: np.ndarray, y: np.ndarray, yl: str, title: str) -> None:
        fig, ax = plt.subplots(figsize=(3.85, 1.65))
        ax.plot(
            x,
            y,
            linestyle="None",
            marker="o",
            markersize=(2 / 3), 
            markerfacecolor="r",
            markeredgecolor="none",
            markeredgewidth=0,
        )
        ax.set_xlabel("time/year", fontsize=8, fontfamily="Times New Roman")
        ax.set_ylabel(yl, fontsize=8, fontfamily="Times New Roman")
        # y 轴标签略上移，避免底部被裁切
        ax.yaxis.set_label_coords(-0.10, 0.55)
        # 标题保持单行：若过长则自动缩小字号，确保不超出右边界
        title_artist = ax.set_title(title, fontsize=9, fontfamily="Times New Roman")
        ax.tick_params(axis="both", labelsize=7)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily("Times New Roman")
        ax.grid(False)
        # 横纵轴范围由数据自适应；适当增大左/下边距避免标签被裁切
        fig.subplots_adjust(left=0.18, right=0.90, bottom=0.24, top=0.88)

        # 保存前做一次渲染，检查标题是否超出图框右边界；若超出则逐步减小字号
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        fig_bbox = fig.get_window_extent(renderer=renderer)
        pad_px = 2.0
        for fs in range(9, 5, -1):  # 9,8,7,6
            title_artist.set_fontsize(fs)
            fig.canvas.draw()
            tb = title_artist.get_window_extent(renderer=renderer)
            if tb.x1 <= fig_bbox.x1 - pad_px:
                break
        png_path = path_png_base.with_suffix(".png")
        fig.savefig(png_path, dpi=600)
        plt.close(fig)
        print(f"图像已保存: {png_path}")

    _plot_series(
        out_dir / f"{base_name}_baseline",
        outN[:, 0],
        outN[:, 1],
        "baseline/mm",
        "Original baseline time series",
    )
    _plot_series(
        out_dir / f"{base_name}_azimuth",
        outN[:, 0],
        outN[:, 2],
        "azimuth/millisecond",
        "Original azimuth time series",
    )
    _plot_series(
        out_dir / f"{base_name}_baseline_detrend",
        outND[:, 0],
        outND[:, 1],
        "baseline/mm",
        f"Baseline time series with linear trend removed (rate = {p1[0]:.2f} mm/a)",
    )
    _plot_series(
        out_dir / f"{base_name}_azimuth_detrend",
        outND[:, 0],
        outND[:, 2],
        "azimuth/millisecond",
        f"Azimuth time series with linear trend removed (rate = {p2[0]:.2f} millisecond/a)",
    )


# ===========================
# 四、程序入口
# ===========================


def main() -> None:
    data_dir = Path(__file__).resolve().parent / "data"
    out_dir = Path(__file__).resolve().parent / "Out"
    out_dir.mkdir(exist_ok=True)

    # 站对列表，可按需增删；文件名需与 data_dir 下的一致
    station_pairs: List[Tuple[str, str]] = [
        ("SCLH_raw.neu", "SCTQ_raw.neu"),
        ("YNYL_raw.neu", "YNYS_raw.neu"),
        ("GSDX_raw.neu", "NXHY_raw.neu"),
    ]

    print(f"数据目录: {data_dir}")
    print(f"输出目录: {out_dir}")
    print(f"共 {len(station_pairs)} 组基线待计算。")

    for i, (f1, f2) in enumerate(station_pairs, start=1):
        print(f"\n--- 正在计算第 {i} 组基线 ---")
        print(f"测站对: {f1} 与 {f2}")
        process_gnss_to_baseline(data_dir, out_dir, f1, f2)
        print(f"✅ 第 {i} 组基线计算完成。")

    print(f"\n所有 {len(station_pairs)} 组基线处理完毕，结果已保存至: {out_dir}")


if __name__ == "__main__":
    main()

