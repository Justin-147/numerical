#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNSS 协调方向异常：时间滤波（20–150 天频带）

功能：
- 读取 GNSS-coordinated-anomaly/DataIn 下 *_raw.neu （CENC的数据格式）
- 构建连续逐日序列（缺测线性插值补齐，满足等间隔采样）
- 20–150 天频带提取二选一（见脚本顶部 DEFAULT_FILTER_MODE）：
  - bandpass：等间隔逐日序列上 Butterworth 带通 + 零相位滤波（sosfiltfilt）
  - hht：EMD + Hilbert 瞬时频率筛选 IMF 叠加
- 输出：
  - FiltDataOut/<SITE>_HHTfilt.txt / <SITE>_Bandfilt.txt：YYYYMMDD, YYYY.DECM, N/E/U_filt(mm), Azimuth(deg)
    （`--filter-mode hht` 时为 HHTfilt；`bandpass` 时为 Bandfilt）
  - FiltDataOut/stinfo.txt：site, lat, lon
  - （可选，默认开启）同名 .png：N/E/U 滤波时间序列三子图
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, medfilt, sosfiltfilt

# ---------------------------------------------------------------------------
# 常数
# ---------------------------------------------------------------------------
# 可修改
DEFAULT_MAX_JOBS = 16 # 最大并行进程数（程序会按站点数自动取 min）
DEFAULT_OUT_PATH = os.path.join("GNSS-coordinated-anomaly", "FiltDataOut") # 输出目录
DEFAULT_VERBOSE = True # 是否打印详细信息
DEFAULT_STRICT_QUALITY = False # 是否严格检查数据质量
# 是否在写出 txt 后绘制 N/E/U 滤波时间序列（上中下三子图）；文件名与 txt 后缀一致（HHTfilt / Bandfilt）
DEFAULT_ENABLE_PLOT = True # 是否绘图

# 20–150 天频带提取方式："bandpass"（Butterworth 带通）或 "hht"（EMD+Hilbert）
# hht相比bandpass细节更多
# bandpass相比hht结果更平滑
DEFAULT_FILTER_MODE = "hht" # 默认使用 hht，如果需要平滑结果，可以改为 bandpass

# 不可修改
F_LO = 1.0 / 150.0 / 86400.0 # 20–150 天周期带通滤波的低频截止频率
F_HI = 1.0 / 20.0 / 86400.0 # 20–150 天周期带通滤波的高频截止频率
DT_SEC = 86400.0 # 一天的秒数
MIN_DAYS_NEU = 365 # .neu 站点最少数据长度要求（1 年 = 365 天）

# 带通滤波器阶数（仅 filter_mode=bandpass 时有效）
BANDPASS_ORDER = 4

# Hilbert 端点效应抑制：对每个 IMF 做镜像延拓的天数（两端各 pad 天），然后裁剪回原长度。
# 越大端点越稳定，但计算略慢；建议 30~90。
HILBERT_MIRROR_PAD_DAYS = 60

# 权重时间平滑窗口（天，必须为奇数；<=1 表示不平滑）
# 直接对每个 IMF 的权重 w(t,k) 平滑，通常比只平滑频率更能压制“高频小抖动”。
WEIGHT_SMOOTH_WIN_DAYS = 11

# 软权重平滑后的小权重截断阈值（半二值化）：w < 阈值 -> 0
# 作用：抑制短周期 IMF 以“小权重漏入”导致的细密高频纹波，同时保留软过渡（阈值以上仍为连续权重）。
WEIGHT_CUTOFF = 0.2

# 最小持续“进带”天数：对每个 IMF 的权重序列，若连续 w>=WEIGHT_CUTOFF 的片段长度 < 该值，则整段置 0。
# 经验：关注 20–150 天游程信号时，可取 12–15（略小于最短周期 20 天）。
MIN_INBAND_RUN_DAYS = 15

# 原始观测预处理：线性趋势拟合 + 3σ 剔除奇异点（对 N/E/U 任一分量超限则剔除整行观测）
ENABLE_TREND_SIGMA_CLIP = True # 是否启用线性趋势拟合 + 3σ 剔除奇异点
TREND_SIGMA_CLIP_K = 3.0 # 3σ 剔除奇异点的倍数
TREND_SIGMA_CLIP_MIN_POINTS = 30 # 最小有效点数，不足时不剔除
ENABLE_TREND_DETREND = True # 是否启用线性趋势拟合 + 扣除趋势


def _output_filt_tag(filter_mode: str) -> str:
    """hht -> HHTfilt；bandpass -> Bandfilt（用于 txt/png 文件名后缀）。"""
    return "HHTfilt" if str(filter_mode).strip().lower() == "hht" else "Bandfilt"


def bandpass_zero_phase_daily(
    x: np.ndarray,
    dt_sec: float,
    f_lo: float,
    f_hi: float,
    order: int = BANDPASS_ORDER,
) -> np.ndarray:
    """
    逐日等间隔序列（采样间隔 dt_sec 秒）上的零相位带通，通带 [f_lo, f_hi]（Hz）。
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = int(x.size)
    if n < 16:
        return np.zeros_like(x)
    fs = 1.0 / float(dt_sec)
    nyq = 0.5 * fs
    lo = f_lo / nyq
    hi = f_hi / nyq
    lo = float(np.clip(lo, 1e-15, 0.999))
    hi = float(np.clip(hi, lo + 1e-15, 0.999999999))
    if lo >= hi:
        return np.zeros_like(x)
    sos = butter(order, [lo, hi], btype="band", output="sos")
    return sosfiltfilt(sos, x)


@dataclass
class StationInfo:
    median_lat: float
    median_lon: float
    site_id: str


def _matlab_if_all_true(cond: np.ndarray) -> bool:
    return bool(np.asarray(cond, dtype=bool).ravel().all())


def _locoma(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float64).ravel()
    l_y = y.size
    x_u: list = []
    y_u: list = []
    for i in range(2, l_y):
        if y[i - 2] < y[i - 1] and y[i - 1] > y[i]:
            x_u.append(i)
            y_u.append(y[i - 1])
        elif y[i - 2] < y[i - 1] and y[i - 1] == y[i]:
            x_u.append(i)
            y_u.append(y[i - 1])
    return np.array(x_u, dtype=np.float64), np.array(y_u, dtype=np.float64)


def _locomi(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float64).ravel()
    l_y = y.size
    x_d: list = []
    y_d: list = []
    for i in range(2, l_y):
        if y[i - 2] > y[i - 1] and y[i - 1] < y[i]:
            x_d.append(i)
            y_d.append(y[i - 1])
        elif y[i - 2] > y[i - 1] and y[i - 1] == y[i]:
            x_d.append(i)
            y_d.append(y[i - 1])
    return np.array(x_d, dtype=np.float64), np.array(y_d, dtype=np.float64)


def _mypredict(
    data_x: np.ndarray,
    data_y: np.ndarray,
    ex_max_x: np.ndarray,
    ex_max_y: np.ndarray,
    ex_min_x: np.ndarray,
    ex_min_y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    l_ma = ex_max_x.size
    l_mi = ex_min_x.size
    _ = (float(np.std(ex_max_y, ddof=1)) if l_ma > 1 else 0.0, float(np.std(ex_min_y, ddof=1)) if l_mi > 1 else 0.0)

    if l_ma >= 3:
        m_maxEX = int(np.round(np.mean(np.diff(ex_max_x[-3:])))) or 1
        p_maxEX1 = ex_max_x[-1] + np.ceil((data_x[-1] - ex_max_x[-1]) / m_maxEX) * m_maxEX
        p_maxEX2 = p_maxEX1 + m_maxEX
        p_maxEX = np.array([p_maxEX1, p_maxEX2], dtype=np.float64)

        m_maxEY = float(np.mean(ex_max_y[-3:]))
        if data_y[-1] > data_y[-2]:
            if m_maxEY > data_y[-1]:
                p_maxEY = np.array([m_maxEY, m_maxEY], dtype=np.float64)
            else:
                m_maxEY = float(np.mean(np.abs(np.diff(ex_max_y[-3:]))))
                p_maxEY = np.array([m_maxEY, m_maxEY], dtype=np.float64) / 2.0 + data_y[-1]
        else:
            p_maxEY = np.array([m_maxEY, m_maxEY], dtype=np.float64)

        m_maxBX = int(np.round(np.mean(np.diff(ex_max_x[:3])))) or 1
        p_maxBX1 = (ex_max_x[0] - data_x[0]) - np.ceil((ex_max_x[0] - data_x[0]) / m_maxBX) * m_maxBX
        p_maxBX2 = p_maxBX1 - m_maxBX
        p_maxBX = np.array([p_maxBX2, p_maxBX1], dtype=np.float64)

        m_maxBY = float(np.mean(ex_max_y[:3]))
        if data_y[0] > data_y[1]:
            if m_maxBY > data_y[0]:
                p_maxBY = np.array([m_maxBY, m_maxBY], dtype=np.float64)
            else:
                m_maxBY = float(np.mean(np.abs(np.diff(ex_max_y[:3]))))
                p_maxBY = np.array([m_maxBY, m_maxBY], dtype=np.float64) / 2.0 + data_y[0]
        else:
            p_maxBY = np.array([m_maxBY, m_maxBY], dtype=np.float64)
    else:
        p_maxEX = np.array([data_x[-1] + 2, data_x[-1] + 4], dtype=np.float64)
        p_maxEY = np.array([ex_max_y[-1], ex_max_y[-1]], dtype=np.float64)
        p_maxBX = np.array([data_x[0] - 4, data_x[0] - 2], dtype=np.float64)
        p_maxBY = np.array([ex_max_y[0], ex_max_y[0]], dtype=np.float64)

    if l_mi >= 3:
        m_minEX = int(np.round(np.mean(np.diff(ex_min_x[-3:])))) or 1
        p_minEX1 = ex_min_x[-1] + np.ceil((data_x[-1] - ex_min_x[-1]) / m_minEX) * m_minEX
        p_minEX2 = p_minEX1 + m_minEX
        p_minEX = np.array([p_minEX1, p_minEX2], dtype=np.float64)

        m_minEY = float(np.mean(ex_min_y[-3:]))
        if data_y[-1] < data_y[-2]:
            if m_minEY < data_y[-1]:
                p_minEY = np.array([m_minEY, m_minEY], dtype=np.float64)
            else:
                m_minEY = float(np.mean(np.abs(np.diff(ex_min_y[-3:]))))
                p_minEY = np.array([m_minEY, m_minEY], dtype=np.float64) / 2.0 + data_y[-1]
        else:
            p_minEY = np.array([m_minEY, m_minEY], dtype=np.float64)

        m_minBX = int(np.round(np.mean(np.diff(ex_min_x[:3])))) or 1
        p_minBX1 = (ex_min_x[0] - data_x[0]) - np.ceil((ex_min_x[0] - data_x[0]) / m_minBX) * m_minBX
        p_minBX2 = p_minBX1 - m_minBX
        p_minBX = np.array([p_minBX2, p_minBX1], dtype=np.float64)

        m_minBY = float(np.mean(ex_min_y[:3]))
        if data_y[0] < data_y[1]:
            if m_minBY < data_y[0]:
                p_minBY = np.array([m_minBY, m_minBY], dtype=np.float64)
            else:
                m_minBY = float(np.mean(np.abs(np.diff(ex_min_y[:3]))))
                p_minBY = np.array([m_minBY, m_minBY], dtype=np.float64) / 2.0 + data_y[0]
        else:
            p_minBY = np.array([m_minBY, m_minBY], dtype=np.float64)
    else:
        p_minEX = np.array([data_x[-1] + 2, data_x[-1] + 4], dtype=np.float64)
        p_minEY = np.array([ex_min_y[-1], ex_min_y[-1]], dtype=np.float64)
        p_minBX = np.array([data_x[0] - 4, data_x[0] - 2], dtype=np.float64)
        p_minBY = np.array([ex_min_y[0], ex_min_y[0]], dtype=np.float64)

    max_x = np.concatenate([p_maxBX, ex_max_x, p_maxEX])
    max_y = np.concatenate([p_maxBY, ex_max_y, p_maxEY])
    min_x = np.concatenate([p_minBX, ex_min_x, p_minEX])
    min_y = np.concatenate([p_minBY, ex_min_y, p_minEY])
    return max_x, max_y, min_x, min_y


def _unique_xy_spline(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if x.size == 0:
        return x, y
    order = np.argsort(x)
    x_s = x[order]
    y_s = y[order]
    ux: list = []
    uy: list = []
    last = None
    for xi, yi in zip(x_s, y_s):
        if last is None or float(xi) > float(last):
            ux.append(float(xi))
            uy.append(float(yi))
            last = float(xi)
    return np.array(ux, dtype=np.float64), np.array(uy, dtype=np.float64)


def _envelopef(
    f_au: np.ndarray,
    f_ad: np.ndarray,
    t_au: np.ndarray,
    t_ad: np.ndarray,
    l_yi: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xq = np.arange(1.0, float(l_yi) + 1.0, 1.0, dtype=np.float64)
    xu, yu = _unique_xy_spline(np.asarray(t_au, dtype=np.float64).ravel(), np.asarray(f_au, dtype=np.float64).ravel())
    xd, yd = _unique_xy_spline(np.asarray(t_ad, dtype=np.float64).ravel(), np.asarray(f_ad, dtype=np.float64).ravel())
    if xu.size < 2 or xd.size < 2:
        raise ValueError("envelopef: 包络点不足")
    cs_u = CubicSpline(xu, yu, bc_type="not-a-knot", extrapolate=True)
    cs_d = CubicSpline(xd, yd, bc_type="not-a-knot", extrapolate=True)
    eyi_u = cs_u(xq)
    eyi_d = cs_d(xq)
    return eyi_u, eyi_d, xq, xq


def _sifting(xi: np.ndarray, yi: np.ndarray, h_n: int) -> np.ndarray:
    yi = np.asarray(yi, dtype=np.float64).copy().ravel()
    l_yi = yi.size
    xi = np.asarray(xi, dtype=np.float64).ravel()
    for _ in range(h_n):
        ex_max_x, ex_max_y = _locoma(yi)
        ex_min_x, ex_min_y = _locomi(yi)
        if ex_max_x.size < 2 or ex_min_x.size < 2:
            break
        max_x, max_y, min_x, min_y = _mypredict(xi, yi, ex_max_x, ex_max_y, ex_min_x, ex_min_y)
        try:
            eyi_u, eyi_d, _, _ = _envelopef(max_y, min_y, max_x, min_x, l_yi)
        except (ValueError, np.linalg.LinAlgError):
            break
        m_y = (eyi_u + eyi_d) / 2.0
        if _matlab_if_all_true(np.abs(m_y) <= 1e-6):
            break
        yi = yi - m_y
    return yi


def emd_to_imf_columns(
    i_n: np.ndarray,
    *,
    e_n: int = 50,
    h_n_e: int = 2000,
) -> np.ndarray:
    """
    直接进行确定性 EMD（无加噪声、无 ensemble）。

    说明：
    - 当前代码里的 `_sifting()` 已经对应 EMD 的“筛分”核心。
    - 本函数按残差逐次提取 IMF（确定性分解，不包含 ensemble 平均或随机噪声）。
    """
    yi = np.asarray(i_n, dtype=np.float64).ravel().copy()
    if yi.size == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if e_n < 1:
        # 与原逻辑保持一致：至少返回 1 列
        return yi.reshape(-1, 1)

    l_yi = yi.size
    xi = np.arange(1, l_yi + 1, dtype=np.float64)

    E = np.zeros((e_n, l_yi), dtype=np.float64)
    j_out = 0
    for j in range(1, e_n + 1):
        imf_e = _sifting(xi, yi, h_n_e)
        E[j - 1, :] = imf_e
        yi = yi - imf_e
        j_out = j
        # 残差已经接近 0 时提前停止
        if float(np.max(np.abs(yi))) <= 1e-12:
            break

    E_out = np.vstack([E[:j_out, :], yi])
    return E_out.T


def parse_raw_neu_file(
    path: str,
) -> Tuple[str, float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    site = ""
    lon = lat = h = 0.0
    rows: List[Tuple[int, float, float, float, float]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#Reference position"):
                parts = line.split()
                if len(parts) >= 6:
                    lon = float(parts[2])
                    lat = float(parts[3])
                    h = float(parts[4])
                    site = parts[5]
                continue
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                ymd = int(parts[0])
                decm = float(parts[1])
                n_mm = float(parts[2])
                e_mm = float(parts[3])
                u_mm = float(parts[4])
            except ValueError:
                continue
            rows.append((ymd, decm, n_mm, e_mm, u_mm))
    if not rows:
        raise ValueError(f"{path} 无有效数据行")
    ymds = np.array([r[0] for r in rows], dtype=np.int32)
    decm = np.array([r[1] for r in rows], dtype=np.float64)
    n = np.array([r[2] for r in rows], dtype=np.float64)
    e = np.array([r[3] for r in rows], dtype=np.float64)
    u = np.array([r[4] for r in rows], dtype=np.float64)
    if not site:
        site = os.path.splitext(os.path.basename(path))[0].replace("_raw", "")
    return site, lon, lat, h, ymds, decm, n, e, u


def _trend_fit_linear(t: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """y ≈ a*t + b（t 建议用 YYYY.DECM；若有效点过少则返回 a=0,b=0）。"""
    t = np.asarray(t, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    ok = np.isfinite(t) & np.isfinite(y)
    if ok.sum() < 2:
        return 0.0, 0.0
    a, b = np.polyfit(t[ok], y[ok], 1)
    return float(a), float(b)


def _trend_sigma_clip_keep_mask(t: np.ndarray, y: np.ndarray, k: float) -> np.ndarray:
    """
    用线性趋势 y≈a*t+b 计算残差 res=y-(a*t+b)，按 |res|<=k*sigma 生成保留掩码。
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    ok = np.isfinite(t) & np.isfinite(y)
    if ok.sum() < int(TREND_SIGMA_CLIP_MIN_POINTS):
        return ok
    a, b = _trend_fit_linear(t, y)
    res = y[ok] - (a * t[ok] + b)
    sigma = float(np.std(res, ddof=1)) if res.size > 1 else 0.0
    if not np.isfinite(sigma) or sigma <= 0.0:
        return ok
    keep_ok = np.abs(res) <= float(k) * sigma
    keep = np.zeros_like(ok, dtype=bool)
    keep[np.flatnonzero(ok)[keep_ok]] = True
    return keep


def _trend_detrend(t: np.ndarray, y: np.ndarray, a: float, b: float) -> np.ndarray:
    """返回 detrend 后的 y：y-(a*t+b)。"""
    t = np.asarray(t, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    out = y.copy()
    ok = np.isfinite(t) & np.isfinite(y)
    out[ok] = y[ok] - (float(a) * t[ok] + float(b))
    return out


def _date_from_yyyymmdd(yyyymmdd: int) -> date:
    s = str(int(yyyymmdd))
    return date(int(s[0:4]), int(s[4:6]), int(s[6:8]))


def build_daily_series_from_neu(
    ymds: np.ndarray,
    values: np.ndarray,
    start_yyyymmdd: Optional[int],
    end_yyyymmdd: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    dts = np.array([_date_from_yyyymmdd(int(x)) for x in ymds], dtype=object)
    d_min = min(dts)  # type: ignore[arg-type]
    d_max = max(dts)  # type: ignore[arg-type]
    if start_yyyymmdd is not None:
        d_min = max(d_min, _date_from_yyyymmdd(start_yyyymmdd))
    if end_yyyymmdd is not None:
        d_max = min(d_max, _date_from_yyyymmdd(end_yyyymmdd))
    if d_max < d_min:
        raise ValueError("指定的时间范围在该站数据中为空")
    n_days = (d_max - d_min).days + 1
    days = np.array([(d_min + timedelta(days=i)).strftime("%Y%m%d") for i in range(n_days)], dtype=np.int32)

    idx = np.array([(_date_from_yyyymmdd(int(x)) - d_min).days for x in ymds], dtype=np.int32)
    in_range = (idx >= 0) & (idx < n_days)
    idx = idx[in_range]
    v = values[in_range]
    series = np.full(n_days, np.nan, dtype=np.float64)
    for i, val in zip(idx, v):
        series[int(i)] = float(val)
    finite = np.isfinite(series)
    if finite.sum() < 2:
        raise ValueError("有效天数过少，无法插值")
    x = np.flatnonzero(finite).astype(np.float64)
    y = series[finite]
    series = np.interp(np.arange(n_days, dtype=np.float64), x, y)
    return days, series


def neu_passes_quality(
    ymds: np.ndarray,
    *,
    start_yyyymmdd: Optional[int],
    end_yyyymmdd: Optional[int],
    doy: Optional[int],
    min_frac: float = 0.95,
) -> bool:
    ymds = np.asarray(ymds)
    if ymds.size == 0:
        return False
    uniq = np.unique(ymds.astype(np.int64))
    if start_yyyymmdd is not None and (int(start_yyyymmdd) not in uniq):
        return False
    if end_yyyymmdd is not None and (int(end_yyyymmdd) not in uniq):
        return False
    if start_yyyymmdd is not None and doy is not None:
        d0 = _date_from_yyyymmdd(int(start_yyyymmdd))
        d1 = d0 + timedelta(days=int(doy) - 1)
        end_expected = int(d1.strftime("%Y%m%d"))
        if end_yyyymmdd is not None and int(end_yyyymmdd) != end_expected:
            return False
        if end_expected not in uniq:
            return False
        n_total = int(doy)
        in_win = (uniq >= int(start_yyyymmdd)) & (uniq <= int(end_expected))
        n_have = int(in_win.sum())
        return n_have >= int(np.floor(n_total * min_frac))
    if start_yyyymmdd is not None and end_yyyymmdd is not None:
        d0 = _date_from_yyyymmdd(int(start_yyyymmdd))
        d1 = _date_from_yyyymmdd(int(end_yyyymmdd))
        if d1 < d0:
            return False
        n_total = (d1 - d0).days + 1
        in_win = (uniq >= int(start_yyyymmdd)) & (uniq <= int(end_yyyymmdd))
        n_have = int(in_win.sum())
        return n_have >= int(np.floor(n_total * min_frac))
    return True


def quantile_clip_interpolate(series: np.ndarray, doy: int, q_lo: float = 0.01, q_hi: float = 0.99) -> np.ndarray:
    qqs = series.astype(np.float64).copy()
    lb = float(np.quantile(qqs, q_lo))
    ub = float(np.quantile(qqs, q_hi))
    mask = (qqs >= lb) & (qqs <= ub)
    idx = np.flatnonzero(mask) + 1
    if idx.size == 0:
        return qqs
    if idx[0] != 1:
        idx = np.concatenate([[1], idx])
    if idx[-1] != doy:
        idx = np.concatenate([idx, [doy]])
    return np.interp(np.arange(1, doy + 1, dtype=np.float64), idx.astype(np.float64), qqs[idx - 1])


def _mirror_pad_1d(x: np.ndarray, pad: int) -> np.ndarray:
    """
    两端镜像延拓（reflect）。pad<=0 时返回原数组。
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = int(x.size)
    if pad <= 0 or n < 2:
        return x
    p = int(min(pad, n - 1))
    left = x[1 : p + 1][::-1]
    right = x[-p - 1 : -1][::-1]
    return np.concatenate([left, x, right], axis=0)


def hilbert_instantaneous(imf_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.signal import hilbert

    n_len, n_all = imf_matrix.shape
    n_imf = n_all - 1
    amp = np.zeros((n_len, n_imf), dtype=np.float64)
    fre = np.zeros((n_len, n_imf), dtype=np.float64)
    for k in range(n_imf):
        col0 = imf_matrix[:, k].astype(np.float64)

        # 端点效应抑制：镜像延拓 -> Hilbert -> 裁剪回原长度
        pad = int(HILBERT_MIRROR_PAD_DAYS)
        col = _mirror_pad_1d(col0, pad)
        p = int((col.size - col0.size) // 2)

        isimf = hilbert(col)
        amp0 = np.abs(isimf)

        # 相位：unwrap(atan2(Im, Re+eps))，显式对“分母”加 eps
        h = np.imag(isimf)
        scale = float(np.nanmax(np.abs(col)))
        eps_eff = 1e-12 * (scale if scale > 0.0 else 1.0)
        th = np.unwrap(np.arctan2(h, col + eps_eff))

        ins = np.diff(th) / (DT_SEC * 2.0 * np.pi)
        fre0 = np.full(col.size, np.nan, dtype=np.float64)
        fre0[:-1] = ins

        # 裁剪回原长度
        amp_cut = amp0[p : p + n_len]
        fre_cut = fre0[p : p + n_len]

        # 与 MATLAB 的 diff 行为保持一致：最后一天没有频率估计
        if fre_cut.size > 0:
            fre_cut[-1] = np.nan

        amp[:, k] = amp_cut
        fre[:, k] = fre_cut
    return imf_matrix[:, :n_imf], amp, fre


def band_limited_component_sum(
    imf_vals: np.ndarray,
    imf_freq: np.ndarray,
    f_lo: float,
    f_hi: float,
) -> np.ndarray:
    n = imf_vals.shape[0]
    m = imf_vals.shape[1]
    out = np.zeros(n, dtype=np.float64)
    W = np.ones((n, m), dtype=np.float64)

    # 1) 频带软权重（逐日）：带内≈1，带外平滑且快速衰减
    bw = float(f_hi - f_lo)
    df = max(1e-12, 0.05 * bw)  # 过渡带宽；越小衰减越快（更接近硬阈值）
    for j in range(n - 1):
        f = imf_freq[j].astype(np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            w_lo = 1.0 / (1.0 + np.exp(-(f - float(f_lo)) / df))
            w_hi = 1.0 / (1.0 + np.exp(-(float(f_hi) - f) / df))
        w = w_lo * w_hi
        # 对 NaN 频率（例如边界）默认不过滤
        w = np.where(np.isfinite(w), w, 1.0)
        W[j, :] = w.astype(np.float64)

    # 最后一天：沿用前一天权重，避免末端突变
    if n >= 2:
        W[-1, :] = W[-2, :]

    # 2) 对权重做时间平滑（比只平滑频率更直接抑制高频小抖动）
    win = int(WEIGHT_SMOOTH_WIN_DAYS)
    if win > 1 and n >= 3:
        if win % 2 == 0:
            win += 1
        for k in range(m):
            W[:, k] = medfilt(W[:, k], kernel_size=win)
        W = np.clip(W, 0.0, 1.0)

    # 3) 半二值化：截断小权重，减少短周期纹波漏入
    cut = float(WEIGHT_CUTOFF)
    if cut > 0.0:
        W = np.where(W >= cut, W, 0.0)

    # 4) 最小持续天数：去掉短暂“误入带”片段（常见于个别时段高频震荡）
    min_run = int(MIN_INBAND_RUN_DAYS)
    if min_run > 1 and n > 0:
        for k in range(m):
            wk = W[:, k]
            active = wk > 0.0
            if not np.any(active):
                continue
            idx = np.flatnonzero(active)
            # 按连续段分组
            starts = [int(idx[0])]
            ends: list[int] = []
            for i in range(1, idx.size):
                if int(idx[i]) != int(idx[i - 1]) + 1:
                    ends.append(int(idx[i - 1]))
                    starts.append(int(idx[i]))
            ends.append(int(idx[-1]))
            # 删除短段
            for s, e in zip(starts, ends):
                if (e - s + 1) < min_run:
                    wk[s : e + 1] = 0.0
            W[:, k] = wk

    # 5) 合成
    out = np.sum(imf_vals.astype(np.float64) * W, axis=1)
    return out


def process_station_file(
    path: str,
    doy: Optional[int],
    imf_number: int = 50,
    shift_time: int = 2000,
    start_date: Optional[int] = None,
    end_date: Optional[int] = None,
    verbose: bool = False,
    strict_quality: bool = False,
    filter_mode: str = DEFAULT_FILTER_MODE,
) -> Optional[Dict[str, Any]]:
    if not path.lower().endswith(".neu"):
        raise ValueError("仅支持 *_raw.neu")

    site, lon, lat, _h, ymds, decm_in, n_mm, e_mm, u_mm = parse_raw_neu_file(path)
    med_lat = float(lat)
    med_lon = float(lon)
    sid = site

    # 先按给定时间范围截取（若不传则用全段）
    # 注意：后续 3σ 剔除可能进一步缩短有效范围；最终输出按剔除后实际范围确定。
    if start_date is not None or end_date is not None:
        lo = int(start_date) if start_date is not None else int(np.nanmin(ymds.astype(np.int64)))
        hi = int(end_date) if end_date is not None else int(np.nanmax(ymds.astype(np.int64)))
        if lo > hi:
            return None
        m = (ymds.astype(np.int64) >= lo) & (ymds.astype(np.int64) <= hi)
        if int(np.sum(m)) < 2:
            return None
        ymds = ymds[m]
        decm_in = decm_in[m]
        n_mm = n_mm[m]
        e_mm = e_mm[m]
        u_mm = u_mm[m]

    # 原始观测奇异点剔除：线性趋势拟合+3σ
    if bool(ENABLE_TREND_SIGMA_CLIP):
        ksig = float(TREND_SIGMA_CLIP_K)
        keep_n = _trend_sigma_clip_keep_mask(decm_in, n_mm, ksig)
        keep_e = _trend_sigma_clip_keep_mask(decm_in, e_mm, ksig)
        keep_u = _trend_sigma_clip_keep_mask(decm_in, u_mm, ksig)
        keep = keep_n & keep_e & keep_u
        if keep.sum() >= 2:
            # 用筛后的观测点拟合趋势，并将“预处理后的（去趋势）数据”进入后续流程
            if bool(ENABLE_TREND_DETREND):
                a_n, b_n = _trend_fit_linear(decm_in[keep], n_mm[keep])
                a_e, b_e = _trend_fit_linear(decm_in[keep], e_mm[keep])
                a_u, b_u = _trend_fit_linear(decm_in[keep], u_mm[keep])
                n_mm = _trend_detrend(decm_in, n_mm, a_n, b_n)
                e_mm = _trend_detrend(decm_in, e_mm, a_e, b_e)
                u_mm = _trend_detrend(decm_in, u_mm, a_u, b_u)
            ymds = ymds[keep]
            decm_in = decm_in[keep]
            n_mm = n_mm[keep]
            e_mm = e_mm[keep]
            u_mm = u_mm[keep]

    if strict_quality:
        if not neu_passes_quality(ymds, start_yyyymmdd=None, end_yyyymmdd=None, doy=doy, min_frac=0.95):
            return None

    # 输出时间范围：仅由剔除后的实际观测日期范围决定（不做外推补齐）
    days, n_series_mm = build_daily_series_from_neu(ymds, n_mm, None, None)
    _, e_series_mm = build_daily_series_from_neu(ymds, e_mm, None, None)
    _, u_series_mm = build_daily_series_from_neu(ymds, u_mm, None, None)
    _, decm_series = build_daily_series_from_neu(ymds, decm_in, None, None)

    n_series = n_series_mm / 1000.0
    e_series = e_series_mm / 1000.0
    u_series = u_series_mm / 1000.0

    if doy is None:
        cur_doy = int(days.size)
        if cur_doy < MIN_DAYS_NEU:
            return None
    else:
        cur_doy = int(doy)
        if int(days.size) < cur_doy:
            return None

    n_series = n_series[:cur_doy]
    e_series = e_series[:cur_doy]
    u_series = u_series[:cur_doy]
    days = days[:cur_doy]
    decm_series = decm_series[:cur_doy]

    mode = str(filter_mode).strip().lower()
    if mode not in ("bandpass", "hht"):
        raise ValueError(f"filter_mode 必须是 'bandpass' 或 'hht'，收到：{filter_mode!r}")

    qqs_n = quantile_clip_interpolate(n_series - n_series[0], cur_doy)
    qqs_e = quantile_clip_interpolate(e_series - e_series[0], cur_doy)
    qqs_u = quantile_clip_interpolate(u_series - u_series[0], cur_doy)

    if mode == "bandpass":
        if verbose:
            print(f"[{site}] bandpass N/E/U：len={cur_doy}, order={BANDPASS_ORDER}, 20-150 d")
        ns_ll = bandpass_zero_phase_daily(qqs_n, DT_SEC, F_LO, F_HI)
        ew_ll = bandpass_zero_phase_daily(qqs_e, DT_SEC, F_LO, F_HI)
        zz_ll = bandpass_zero_phase_daily(qqs_u, DT_SEC, F_LO, F_HI)
    else:
        if verbose:
            print(f"[{site}] EMD(N) 开始：len={cur_doy}, imf_number={imf_number}, shift_time={shift_time}")
        imf_n_full = emd_to_imf_columns(qqs_n, e_n=imf_number, h_n_e=shift_time)
        imf_n, _ns_amp, ns_fre = hilbert_instantaneous(imf_n_full)
        ns_ll = band_limited_component_sum(imf_n, ns_fre, F_LO, F_HI)

        if verbose:
            print(f"[{sid}] EMD(E) 开始")
        imf_e_full = emd_to_imf_columns(qqs_e, e_n=imf_number, h_n_e=shift_time)
        imf_e, _ew_amp, ew_fre = hilbert_instantaneous(imf_e_full)
        ew_ll = band_limited_component_sum(imf_e, ew_fre, F_LO, F_HI)

        if verbose:
            print(f"[{sid}] EMD(U) 开始")
        imf_u_full = emd_to_imf_columns(qqs_u, e_n=imf_number, h_n_e=shift_time)
        imf_u, _z_amp, z_fre = hilbert_instantaneous(imf_u_full)
        zz_ll = band_limited_component_sum(imf_u, z_fre, F_LO, F_HI)

    aa = np.full(cur_doy, np.nan)
    for d in range(cur_doy):
        n0 = float(ns_ll[d])
        e0 = float(ew_ll[d])
        if n0 == 0.0 and e0 == 0.0:
            continue
        if e0 == 0.0:
            aa[d] = 0.0 if n0 >= 0.0 else 180.0
            continue
        ang = float(np.arctan2(e0, n0))
        if ang < 0.0:
            ang += 2.0 * np.pi
        aa[d] = float(np.degrees(ang))

    return {
        "site_id": sid,
        "days_yyyymmdd": days,
        "days_decm": decm_series,
        "station_info": StationInfo(med_lat, med_lon, sid),
        "AA": aa,
        "NS_ll": ns_ll,
        "EW_ll": ew_ll,
        "ZZ_ll": zz_ll,
        "filter_mode": mode,
    }


def _plot_station_hht_series(out_path: str, pack: Dict[str, Any]) -> str:
    """N/E/U 滤波（mm）时间序列，三子图；保存为 <SITE>_HHTfilt.png 或 <SITE>_Bandfilt.png。"""
    from datetime import datetime

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    sid = str(pack["site_id"])
    days = np.asarray(pack["days_yyyymmdd"], dtype=np.int64)
    n_mm = np.asarray(pack["NS_ll"], dtype=np.float64) * 1000.0
    e_mm = np.asarray(pack["EW_ll"], dtype=np.float64) * 1000.0
    u_mm = np.asarray(pack["ZZ_ll"], dtype=np.float64) * 1000.0
    x = [datetime.strptime(str(int(d)), "%Y%m%d") for d in days]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["N_filt (mm)", "E_filt (mm)", "U_filt (mm)"]
    for ax, y, lab in zip(axes, (n_mm, e_mm, u_mm), labels):
        ax.plot(x, y, lw=0.85, color="C0")
        ax.set_ylabel(lab)
        ax.axhline(0.0, color="gray", lw=0.5, ls="--")
    fm = str(pack.get("filter_mode", "hht")).lower()
    if fm == "bandpass":
        axes[0].set_title(f"{sid} N/E/U filtered (Bandpass, 20–150 d, order={BANDPASS_ORDER})")
    else:
        axes[0].set_title(f"{sid} N/E/U filtered (HHT, 20–150 d)")
    axes[-1].set_xlabel("Date")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    tag = _output_filt_tag(str(pack.get("filter_mode", "hht")))
    out_png = os.path.join(out_path, f"{sid}_{tag}.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def save_station_txt(out_path: str, pack: Dict[str, Any]) -> None:
    sid = str(pack["site_id"])
    tag = _output_filt_tag(str(pack.get("filter_mode", "hht")))
    fname = os.path.join(out_path, f"{sid}_{tag}.txt")
    days = np.asarray(pack["days_yyyymmdd"], dtype=np.int32)
    decm = np.asarray(pack["days_decm"], dtype=np.float64)
    ns = np.asarray(pack["NS_ll"], dtype=np.float64) * 1000.0
    ew = np.asarray(pack["EW_ll"], dtype=np.float64) * 1000.0
    zz = np.asarray(pack["ZZ_ll"], dtype=np.float64) * 1000.0
    aa = np.asarray(pack["AA"], dtype=np.float64)
    arr = np.column_stack([days, decm, ns, ew, zz, aa])
    header = "YYYYMMDD\tYYYY.DECM\tN_filt(mm)\tE_filt(mm)\tU_filt(mm)\tAzimuth(deg)"
    np.savetxt(fname, arr, fmt=["%d", "%.4f", "%.10g", "%.10g", "%.10g", "%.10g"], delimiter="\t", header=header, comments="# ")


def save_stinfo(out_path: str, stations: List[StationInfo]) -> None:
    p = os.path.join(out_path, "stinfo.txt")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write("site\tlat\tlon\n")
        for s in stations:
            f.write(f"{s.site_id}\t{s.median_lat:.6f}\t{s.median_lon:.6f}\n")


def _process_and_save_one_station(
    path: str,
    out_path: str,
    *,
    doy: Optional[int],
    imf_number: int,
    shift_time: int,
    start_date: Optional[int],
    end_date: Optional[int],
    verbose: bool,
    strict_quality: bool,
    enable_plot: bool,
    filter_mode: str,
) -> Optional[StationInfo]:
    pack = process_station_file(
        path,
        doy,
        imf_number=imf_number,
        shift_time=shift_time,
        start_date=start_date,
        end_date=end_date,
        verbose=verbose,
        strict_quality=strict_quality,
        filter_mode=filter_mode,
    )
    if pack is None:
        return None
    os.makedirs(out_path, exist_ok=True)
    save_station_txt(out_path, pack)
    if enable_plot:
        _plot_station_hht_series(out_path, pack)
    return pack["station_info"]


def run_station_batch(args: argparse.Namespace) -> None:
    files = sorted(glob.glob(os.path.join(args.data_path, args.glob_pattern)))
    if not files:
        print(f"未找到输入文件：{args.data_path} / {args.glob_pattern}", file=sys.stderr)
        sys.exit(1)

    jobs = min(int(args.max_jobs), len(files))
    if args.verbose:
        print(f"并行设置：max_jobs={args.max_jobs}, 站点数={len(files)} -> jobs={jobs}")

    stations: List[StationInfo] = []
    out_tag = _output_filt_tag(str(args.filter_mode))
    if jobs <= 1:
        for path in files:
            if args.verbose:
                print(f"开始处理 {os.path.basename(path)} ...")
            t0 = time.perf_counter()
            st = _process_and_save_one_station(
                path,
                args.out_path,
                doy=args.doy,
                imf_number=args.imf_number,
                shift_time=args.shift_time,
                start_date=args.start_date,
                end_date=args.end_date,
                verbose=False,
                strict_quality=bool(args.strict_quality),
                enable_plot=bool(args.enable_plot),
                filter_mode=str(args.filter_mode),
            )
            if st is not None:
                stations.append(st)
                dt = time.perf_counter() - t0
                print(f"完成 {st.site_id}: {st.site_id}_{out_tag}.txt（{dt:.1f}s）")
    else:
        with cf.ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = [
                ex.submit(
                    _process_and_save_one_station,
                    path,
                    args.out_path,
                    doy=args.doy,
                    imf_number=args.imf_number,
                    shift_time=args.shift_time,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    verbose=False,
                    strict_quality=bool(args.strict_quality),
                    enable_plot=bool(args.enable_plot),
                    filter_mode=str(args.filter_mode),
                )
                for path in files
            ]
            done = 0
            total = len(futs)
            t0 = time.perf_counter()
            for fut in cf.as_completed(futs):
                st = fut.result()
                if st is None:
                    continue
                stations.append(st)
                done += 1
                elapsed = time.perf_counter() - t0
                print(f"[{done}/{total}] 完成 {st.site_id} -> {st.site_id}_{out_tag}.txt（累计 {elapsed/60:.1f} min）")

    save_stinfo(args.out_path, stations)
    print(f"共处理 {len(stations)} 个站，stinfo 已写入 {args.out_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GNSS 协调方向异常：时间滤波（20–150 天，HHT 或带通）")
    ps = p.add_argument_group("stations")
    ps.add_argument("--data-path", default=os.path.join("GNSS-coordinated-anomaly", "DataIn"))
    ps.add_argument("--out-path", default=DEFAULT_OUT_PATH)
    ps.add_argument("--glob-pattern", default="*_raw.neu")
    ps.add_argument("--start-date", type=int, default=None, help="起始日期 YYYYMMDD（含）")
    ps.add_argument("--end-date", type=int, default=None, help="结束日期 YYYYMMDD（含）")
    ps.add_argument("--doy", type=int, default=None, help="截取前 doy 天；None 表示用完整范围")
    ps.add_argument("--strict-quality", action="store_true", default=DEFAULT_STRICT_QUALITY)
    ps.add_argument("--imf-number", type=int, default=50)
    ps.add_argument("--shift-time", type=int, default=2000)
    ps.add_argument("--max-jobs", type=int, default=DEFAULT_MAX_JOBS)
    ps.add_argument("--verbose", action="store_true", default=DEFAULT_VERBOSE)
    ps.add_argument(
        "--no-plot",
        action="store_true",
        help="关闭 HHT 时间序列图（默认按脚本顶部 DEFAULT_ENABLE_PLOT，且默认开启）",
    )
    ps.add_argument(
        "--filter-mode",
        choices=("bandpass", "hht"),
        default=DEFAULT_FILTER_MODE,
        help="20–150 天提取方式：bandpass=Butterworth 零相位带通（默认）；hht=EMD+Hilbert IMF 叠加",
    )
    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    args.enable_plot = bool(DEFAULT_ENABLE_PLOT) and not bool(getattr(args, "no_plot", False))
    run_station_batch(args)


if __name__ == "__main__":
    main()

