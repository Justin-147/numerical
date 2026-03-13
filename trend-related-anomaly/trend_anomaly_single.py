# -*- coding: utf-8 -*-
"""
单测项趋势转折（速率变化）预报效能分析。

流程概要：
- 读取观测数据（2 列或 25 列日均/整点），时间连续补齐；缺数插值、台阶处理；
- 归一化后滑动窗口左右线性拟合求矢量转角，取峰得到角度极值序列；
- 在阈值×预报期网格上计算 R 值，剔除落在干扰时段内的异常点；
- 有震例时输出 Rmax、R0、报准/漏报、结果文件与图件；无震例时仅提示结束。

输入：支持 2 列日值（8 位 yyyymmdd）、2 列整点值（10 位 yyyymmddhh，程序内转日均）、或多列（日期+24 小时值取日均）。异常判识为矢量转角超过阈值；R > R0 表示具有预测意义。
"""

import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional

warnings.filterwarnings("ignore", category=UserWarning, message=".*[Ll]evel value of.*too high.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*[Dd]iscarding nonzero nanoseconds.*")

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 内部时间历元：儒略日，1970-01-01 对应 719529
_MATLAB_DATENUM_1970 = 719529


def _to_matlab_datenum(ts: pd.Timestamp) -> float:
    delta = (ts - pd.Timestamp("1970-01-01")).total_seconds() / 86400.0
    return _MATLAB_DATENUM_1970 + delta


def _from_matlab_datenum(d: float) -> pd.Timestamp:
    return pd.Timestamp("1970-01-01") + pd.Timedelta(days=float(d) - _MATLAB_DATENUM_1970)


def _datetime_like_to_julian(arr) -> np.ndarray:
    dt = pd.to_datetime(arr)
    if pd.api.types.is_scalar(dt):
        return np.array([_to_matlab_datenum(pd.Timestamp(dt))], dtype=float)
    return np.asarray([_to_matlab_datenum(pd.Timestamp(t)) for t in dt], dtype=float)


def _parse_time_code_to_timestamp(code) -> pd.Timestamp:
    s = str(int(float(code)))
    if len(s) == 14:
        year, month = int(s[0:4]), int(s[4:6])
        day, hour = int(s[6:8]), int(s[8:10])
        minute, second = int(s[10:12]), int(s[12:14])
    elif len(s) == 12:
        year, month = int(s[0:4]), int(s[4:6])
        day, hour = int(s[6:8]), int(s[8:10])
        minute, second = int(s[10:12]), 0
    elif len(s) == 10:
        year, month = int(s[0:4]), int(s[4:6])
        day, hour = int(s[6:8]), int(s[8:10])
        minute, second = 0, 0
    elif len(s) == 8:
        year, month, day = int(s[0:4]), int(s[4:6]), int(s[6:8])
        hour, minute, second = 0, 0, 0
    elif len(s) == 6:
        year, month = int(s[0:4]), int(s[4:6])
        day, hour, minute, second = 1, 0, 0, 0
    elif len(s) == 4:
        year = int(s[0:4])
        month, day, hour, minute, second = 1, 1, 0, 0, 0
    else:
        return pd.to_datetime(s)
    return pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute, second=second)


def _to_datetime_from_julian(times: np.ndarray) -> np.ndarray:
    return np.array([_from_matlab_datenum(t).to_pydatetime() for t in np.atleast_1d(times).ravel()])


# ===========================
# 一、地震目录读取与筛选
# ===========================


def points_in_polygon(xs: np.ndarray, ys: np.ndarray, poly_x: np.ndarray, poly_y: np.ndarray) -> np.ndarray:
    n_points = xs.shape[0]
    inside = np.zeros(n_points, dtype=bool)
    n_vert = len(poly_x)
    for i in range(n_points):
        x, y = xs[i], ys[i]
        j = n_vert - 1
        c = False
        for k in range(n_vert):
            if ((poly_y[k] > y) != (poly_y[j] > y)) and (
                x < (poly_x[j] - poly_x[k]) * (y - poly_y[k]) / (poly_y[j] - poly_y[k] + 1e-12) + poly_x[k]
            ):
                c = not c
            j = k
        inside[i] = c
    return inside


class EarthquakeCatalog:
    """
    地震目录读取与筛选：支持 EQT 定长格式。
    按定长列解析，不 strip 行内容以免破坏列位置；解析失败的行跳过。
    """

    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
        self._df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df
        lines = self.file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        records: List[dict] = []
        for line in lines:
            if not line.strip():
                continue
            if len(line) < 35:
                continue
            try:
                year = int(line[1:5])
                month = int(line[5:7])
                day = int(line[7:9])
                hour = int(line[9:11])
                minute = int(line[11:13])
                second = int(line[13:15])
                lat = float(line[16:21])
                lon = float(line[22:28])
                mag = float(line[28:31])
                depth = float(line[33:35])
            except (ValueError, IndexError):
                continue
            records.append({
                "time": pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute, second=second),
                "lon": lon,
                "lat": lat,
                "mag": mag,
                "depth": depth,
            })
        self._df = pd.DataFrame.from_records(records)
        return self._df

    def select(
        self,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        mag_min: float,
        mag_max: float,
        polygon_lon: np.ndarray,
        polygon_lat: np.ndarray,
    ) -> pd.DataFrame:
        df = self.load()
        mask_time = (df["time"] >= start_time) & (df["time"] <= end_time)
        mask_mag = (df["mag"] >= mag_min) & (df["mag"] <= mag_max)
        inside = points_in_polygon(df["lon"].to_numpy(), df["lat"].to_numpy(), polygon_lon, polygon_lat)
        return df.loc[mask_time & mask_mag & inside].reset_index(drop=True)


# ===========================
# 二、观测数据读取与预处理
# ===========================


def _time_code_to_datenum(time_list: np.ndarray) -> np.ndarray:
    """将时间码（yyyymmdd 等）转为儒略日数。"""
    out = []
    for t in np.atleast_1d(time_list).ravel():
        ts = _parse_time_code_to_timestamp(t)
        out.append(_to_matlab_datenum(ts))
    return np.array(out, dtype=float)


def read_data(data_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取观测数据并补齐断数。
    - 2 列：第1列时间码，第2列观测值。
      - 8 位（yyyymmdd）：日值，补齐空缺使时间连续，缺日填 999999，返回 (datenum, 观测值)。
      - 10 位（yyyymmddhh）：整点值，原样返回 (时间码 int64, 观测值)，由主流程转为日均值。
    - 多于 2 列：第1列日期（yyyymmdd），第2–25 列为 24 小时值，取日均值后按日补齐。
    """
    data_path = Path(data_path)
    if not data_path.exists():
        return np.array([]), np.array([])
    try:
        raw = np.loadtxt(data_path, dtype=np.float64, ndmin=2)
    except Exception:
        try:
            with open(data_path, encoding="utf-8", errors="ignore") as f:
                lines = [ln for ln in f if ln.strip() and not ln.strip().startswith("#")]
        except Exception:
            return np.array([]), np.array([])
        if not lines:
            return np.array([]), np.array([])
        try:
            raw = np.loadtxt(lines, dtype=np.float64, ndmin=2)
        except Exception:
            return np.array([]), np.array([])
    if raw.size == 0:
        return np.array([]), np.array([])

    m, n = raw.shape
    if n == 2:
        time_codes = raw[:, 0]
        obs = raw[:, 1].astype(float)
    else:
        time_codes = raw[:, 0]
        obs = np.mean(raw[:, 1 : min(25, n)], axis=1).astype(float)

    time_codes = np.asarray(time_codes, dtype=np.float64)
    if len(time_codes) == 0:
        return np.array([]), np.array([])

    time_len = len(str(int(time_codes[0])))
    if n == 2 and time_len == 10:
        # 整点值（yyyymmddhh）：不转 datenum，直接返回时间码与观测值，供主流程转日均
        return time_codes.astype(np.int64), np.asarray(obs, dtype=float)
    time_datenum = _time_code_to_datenum(time_codes)
    if time_len == 8:
        t_min, t_max = time_datenum.min(), time_datenum.max()
        n_days = int(round(t_max - t_min)) + 1
        time_cont = np.linspace(t_min, t_max, n_days)
        time_cont = np.round(time_cont * 1e6) / 1e6
        obs_filled = np.full(n_days, 999999.0, dtype=float)
        for i, t in enumerate(time_datenum):
            idx = int(round(t - t_min))
            if 0 <= idx < n_days:
                obs_filled[idx] = obs[i]
        return time_cont, obs_filled
    return time_datenum, np.asarray(obs, dtype=float)


def data_preprocess(obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    缺数插值 + 台阶处理。
    返回 (预处理后序列, 缺数位置索引 mi_P)，用于 windows_length<90 时将缺数日并入干扰时段。
    """
    obs = np.asarray(obs, dtype=float).copy()
    missing = (obs == 999999.0) | (obs == 99999.0)
    mi_P = np.where(missing)[0]
    valid_idx = np.where(~missing)[0]
    if len(valid_idx) == 0:
        return obs, mi_P
    if len(mi_P) > 0:
        f = interp1d(valid_idx, obs[valid_idx], kind="linear", bounds_error=False, fill_value="extrapolate")
        obs[mi_P] = f(mi_P)
    dif = np.diff(obs)
    if len(dif) == 0:
        return obs, mi_P
    mean_square = 3 * np.std(dif, ddof=1) + np.mean(dif)
    for i in range(1, len(obs)):
        d = dif[i - 1]
        if abs(d) > abs(mean_square) and d > 0:
            obs[i:] -= d
        elif abs(d) > abs(mean_square) and d < 0:
            obs[i:] += abs(d)
    return obs, mi_P


def _interpolate_missing_pchip_only(obs: np.ndarray, missing_val: float = 999999.0) -> np.ndarray:
    """仅对缺测（999999/99999）做 PCHIP 插值，不裁剪、不台阶。用于整点值转日均时的缺数插值。"""
    obs = np.asarray(obs, dtype=float).copy()
    missing_mask = (obs == 999999.0) | (obs == 99999.0)
    valid_idx = np.where(~missing_mask)[0]
    if valid_idx.size == 0:
        return obs
    mi_P = np.where(missing_mask)[0]
    m_P = valid_idx
    if len(m_P) >= 2:
        pchip = PchipInterpolator(m_P, obs[m_P])
        obs[mi_P] = pchip(mi_P)
    else:
        obs[mi_P] = np.interp(mi_P, m_P, obs[m_P])
    return obs


def complete_days(first_day: int, last_day: int) -> List[int]:
    """根据首尾 yyyymmdd 生成连续日期列表。"""
    start = pd.to_datetime(str(first_day), format="%Y%m%d")
    end = pd.to_datetime(str(last_day), format="%Y%m%d")
    dates = pd.date_range(start, end, freq="D")
    return [int(d.strftime("%Y%m%d")) for d in dates]


def preprocess_missing_and_steps_hourly(obs: np.ndarray) -> np.ndarray:
    """
    整点值缺数插值 + 台阶修正。
    缺测 999999/99999 用 PCHIP 插值；开端大量缺数时裁剪；台阶用 3×std(diff) 判定并后段平移。
    """
    obs = np.asarray(obs, dtype=float).copy()
    missing_mask = (obs == 999999.0) | (obs == 99999.0)
    valid_idx = np.where(~missing_mask)[0]
    if valid_idx.size == 0:
        return obs
    mi_P = np.where(missing_mask)[0]
    m_P = valid_idx
    if len(m_P) >= 2:
        pchip = PchipInterpolator(m_P, obs[m_P])
        obs[mi_P] = pchip(mi_P)
    else:
        obs[mi_P] = np.interp(mi_P, m_P, obs[m_P])
    if mi_P.size > 0 and mi_P[0] == 0:
        first_valid = int(valid_idx[0])
        if first_valid > 24:
            obs = obs[first_valid + 1 :]
        else:
            obs = obs[24:]
    dif = np.diff(obs)
    if len(dif) == 0:
        return obs
    mean_square = 3 * np.std(dif, ddof=1)
    for i in range(1, len(obs)):
        d = dif[i - 1]
        if abs(d) > abs(mean_square) and d > 0:
            obs[i:] -= d
        elif abs(d) > abs(mean_square) and d < 0:
            obs[i:] += abs(d)
    return obs


def daily_mean_from_hourly(
    time_labels: np.ndarray, data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    整点值求日均值。时间码为 10 位 yyyymmddhh；起点非 00 时裁前若干小时；缺日时补全后 PCHIP 插值再按 24 点取均值。
    """
    time_labels = np.asarray(time_labels, dtype=np.int64)
    data = np.asarray(data, dtype=float)
    if time_labels[0] % 100 != 0:
        n_drop = min(24 - (time_labels[0] % 100), len(time_labels))
        time_labels = time_labels[n_drop:]
        data = data[n_drop:]
    if len(time_labels) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=float)
    first_day = int(time_labels[0] // 100)
    last_day = int(time_labels[-1] // 100)
    all_days = complete_days(first_day, last_day)
    N2 = len(all_days)
    N1 = len(time_labels) // 24
    if N1 < N2:
        time_full: List[int] = []
        for d in all_days:
            for h in range(24):
                time_full.append(d * 100 + h)
        time_full_arr = np.array(time_full, dtype=np.int64)
        data_full = np.full(len(time_full_arr), 999999.0, dtype=float)
        t_to_val: Dict[int, float] = dict(zip(time_labels.tolist(), data.tolist()))
        for i, t in enumerate(time_full_arr):
            if t in t_to_val:
                data_full[i] = t_to_val[t]
        data_full = _interpolate_missing_pchip_only(data_full)
        n_blocks = len(time_full_arr) // 24
        out_days = (time_full_arr[::24] // 100).astype(np.int64)
        out_means = np.array(
            [float(np.nanmean(data_full[j : j + 24])) for j in range(0, n_blocks * 24, 24)],
            dtype=float,
        )
        return out_days, out_means
    n_blocks = len(time_labels) // 24
    if n_blocks == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=float)
    use_len = n_blocks * 24
    out_days = np.array(
        [int(time_labels[j] // 100) for j in range(0, use_len, 24)], dtype=np.int64
    )
    block_vals = data[:use_len].reshape(n_blocks, 24).astype(float)
    block_vals = np.where(
        (block_vals == 999999.0) | (block_vals == 99999.0), np.nan, block_vals
    )
    out_means = np.nanmean(block_vals, axis=1).astype(float)
    return out_days, out_means


# ===========================
# 三、归一化与矢量转角
# ===========================


def normalization_01(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
    d_min = float(np.min(data))
    d_max = float(np.max(data))
    if d_max == d_min:
        return np.zeros_like(data, dtype=float), d_min, d_max
    return (data - d_min) / (d_max - d_min), d_min, d_max


def mean_square_background2(
    time_date: np.ndarray,
    pre_observed: np.ndarray,
    windows_min_period: int,
    windows_step: int = 1,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """
    给定窗口的全局转折角度时序（矢量转角）。
    windows_step：滑动步长（天），窗口每次移动的天数。
    返回：ALLAngle, error, Mean_value, MeanSquareBackgroundData(标准差), ALLAngle_max。
    """
    date_norm, _, _ = normalization_01(time_date.astype(float))
    data_norm, _, _ = normalization_01(pre_observed.astype(float))
    n = len(time_date)
    w = windows_min_period
    step = max(1, int(windows_step))
    all_angle: List[float] = []
    error1 = np.full(n, np.nan, dtype=float)
    error2 = np.full(n, np.nan, dtype=float)

    for i in range(w, n - w, step):
        x1 = date_norm[i - w : i + 1]
        y1 = data_norm[i - w : i + 1]
        p1 = np.polyfit(x1, y1, 1)
        fit1 = np.polyval(p1, x1)
        error1[i] = np.sqrt(np.mean((y1 - fit1) ** 2))
        n1 = np.array([date_norm[i] - date_norm[i - w], fit1[-1] - fit1[0]], dtype=float)

        x2 = date_norm[i : i + w + 1]
        y2 = data_norm[i : i + w + 1]
        p2 = np.polyfit(x2, y2, 1)
        fit2 = np.polyval(p2, x2)
        error2[i] = np.sqrt(np.mean((y2 - fit2) ** 2))
        n2 = np.array([date_norm[i + w] - date_norm[i], fit2[-1] - fit2[0]], dtype=float)

        dot_n = np.dot(n1, n2)
        norm_prod = np.linalg.norm(n1) * np.linalg.norm(n2)
        if norm_prod < 1e-12:
            angle_rad = 0.0
        else:
            cos_a = np.clip(dot_n / norm_prod, -1.0, 1.0)
            angle_rad = np.arccos(cos_a)
        all_angle.append(float(angle_rad))

    all_angle = np.array(all_angle, dtype=float) * (180.0 / np.pi)
    all_angle = np.real(all_angle)
    indices = list(range(w, n - w, step))
    full_angle = np.zeros(n, dtype=float)
    for j, i in enumerate(indices):
        full_angle[i] = all_angle[j]

    err = np.nansum(np.stack([np.nan_to_num(error1), np.nan_to_num(error2)], axis=0), axis=0)

    min_dist = max(1, w // 2)
    peaks, _ = find_peaks(full_angle, distance=min_dist)
    all_angle_max = np.zeros(n, dtype=float)
    all_angle_max[peaks] = full_angle[peaks]

    non_zero = all_angle_max[all_angle_max != 0]
    mean_value = float(np.mean(non_zero)) if len(non_zero) > 0 else 0.0
    std_value = float(np.std(non_zero, ddof=1)) if len(non_zero) > 1 else 0.0
    return full_angle, err, mean_value, std_value, all_angle_max  # ALLAngle, error, Mean_value, Std_value, ALLAngle_max


# ===========================
# 四、干扰剔除与 R 值计算
# ===========================


def del_interference(
    abnormal_times: np.ndarray,
    st_t: np.ndarray,
    ed_t: np.ndarray,
) -> np.ndarray:
    """剔除落在干扰时段 [stT, edT] 内的异常时间点。"""
    if len(st_t) == 0 or len(ed_t) == 0:
        return abnormal_times
    keep = np.ones(len(abnormal_times), dtype=bool)
    for i, t in enumerate(abnormal_times):
        for j in range(len(st_t)):
            if st_t[j] <= t <= ed_t[j]:
                keep[i] = False
                break
    return abnormal_times[keep]


def is_predicted(
    eq_times: np.ndarray,
    ab_st: np.ndarray,
    ab_ed: np.ndarray,
    alarm_day: int,
) -> np.ndarray:
    """
    报准判定：与 R 程序一致。判断每个地震是否落在任一异常段的预报窗内。
    预报窗规则：对每段 [st, ed]，若段长 > 预报期则预报窗为 [st, ed]，否则为 [st, st+alarm_day]。
    判定方式：地震时间取整到天 floor(eq)，落在任意一段预报窗 [st, ed] 内即视为报准。
    """
    pred_intervals: List[Tuple[float, float]] = []
    for st, ed in zip(np.atleast_1d(ab_st).ravel(), np.atleast_1d(ab_ed).ravel()):
        st, ed = float(st), float(ed)
        if ed - st > alarm_day:
            pred_intervals.append((st, ed))
        else:
            pred_intervals.append((st, st + alarm_day))
    flags = []
    for t_eq in eq_times:
        t_eq_int = float(np.floor(t_eq))
        ok = any(st <= t_eq_int <= ed for st, ed in pred_intervals)
        flags.append(ok)
    return np.asarray(flags, dtype=bool)


def r_value_single(
    eq_times: np.ndarray,
    alarm_start: np.ndarray,
    alarm_sec: np.ndarray,
    alarm_days: int,
    dt_start: float,
    dt_end: float,
) -> Tuple[float, float, float]:
    """
    针对单一预测期计算 R 值（与 R 程序一致）：R = 报准率 - 时间占用率。
    报准率 = 报准地震数/地震总数；时间占用率 = 各段预报窗长度之和/总时间跨度。
    预报窗结束 = 段开始 + max(段长, alarm_days)，再按 is_predicted 做报准判定（整数日）。
    """
    n_eq = max(len(eq_times), 1)
    dt_span = max(float(dt_end - dt_start), 1e-6)
    if len(alarm_start) == 0:
        return 0.0, 0.0, 0.0
    ab_ed = np.array(
        [st + (sec if sec > alarm_days else alarm_days) for st, sec in zip(alarm_start, alarm_sec)],
        dtype=float,
    )
    flags = is_predicted(eq_times, alarm_start, ab_ed, alarm_days)
    success = int(np.sum(flags))
    success_rate = success / n_eq
    total_len = float(np.sum(ab_ed - alarm_start))
    occupied_rate = total_len / dt_span
    return success_rate - occupied_rate, success_rate, occupied_rate


def get_r0(success_count: int, miss_count: int, alpha: float = 0.025) -> float:
    """
    由报准数与漏报数计算 R0 阈值（累积二项分布近似）。
    """
    k = success_count
    n = success_count + miss_count
    if n == 0:
        return 0.0
    min_alpha = 1e38
    min_alpha_p = 0.0
    from math import comb
    for p in np.linspace(0.0, 1.0, 1001):
        s = sum(comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k, n + 1))
        diff = abs(s - alpha)
        if diff < min_alpha:
            min_alpha = diff
            min_alpha_p = p
    return k / n - min_alpha_p


def rtt(
    time_date: np.ndarray,
    pre_observed: np.ndarray,
    eq_dt_arr: np.ndarray,
    st_t: np.ndarray,
    ed_t: np.ndarray,
    day_e: int,
    windows_length: int,
    windows_step: int = 1,
    thres_s: int = 50,
    thres_n: int = 1,
    thres_e: int = 180,
    day_s: int = 1,
    day_n: int = 1,
) -> Tuple[
    np.ndarray,
    List[np.ndarray],
    List[np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
]:
    """
    在阈值×预报期网格上循环计算 R 值，并得到各阈值下的异常时间与角度。
    返回：R, abnorm_X, abnorm_Y, alam_days_arr, threshold_series, ALLAngle, error, ALLAngle_max, Mean_value, Std_value。
    """
    all_angle, error, mean_value, std_value, all_angle_max = mean_square_background2(
        time_date, pre_observed, windows_length, windows_step
    )
    # 异常点：ALLAngle_max != 0 的时间
    ab_time_indices = np.where(all_angle_max != 0)[0]
    ab_times_raw = time_date[ab_time_indices]
    feature_points_time = del_interference(ab_times_raw, st_t, ed_t)
    # 从 all_angle_max 中把被剔除的异常点置 0
    keep_mask = np.isin(ab_times_raw, feature_points_time)
    drop_indices = ab_time_indices[~keep_mask]
    all_angle_max_out = all_angle_max.copy()
    all_angle_max_out[drop_indices] = 0.0

    threshold_series = np.arange(thres_s, thres_e + 1e-9, thres_n, dtype=float)
    alam_days_arr = np.arange(day_s, day_e + 1, day_n, dtype=int)
    dt_start = float(time_date[0])
    dt_end = float(time_date[-1])

    R_all: List[np.ndarray] = []
    abnorm_x_list: List[np.ndarray] = []
    abnorm_y_list: List[np.ndarray] = []

    for thresh in threshold_series:
        above = all_angle_max_out > thresh
        if np.any(above):
            abnormal_time = time_date[above]
            abnormal_data = pre_observed[above]
        else:
            abnormal_time = np.array([], dtype=float)
            abnormal_data = np.array([], dtype=float)
        abnorm_x_list.append(abnormal_time)
        abnorm_y_list.append(abnormal_data)
        alarm_sec = np.ones(len(abnormal_time), dtype=float)
        r_row = []
        for alam_days in alam_days_arr:
            r_val, _, _ = r_value_single(
                eq_dt_arr, abnormal_time, alarm_sec, int(alam_days), dt_start, dt_end
            )
            r_row.append(r_val)
        R_all.append(np.array(r_row, dtype=float))
    R = np.array(R_all, dtype=float)
    R = np.maximum(R, -1.0)  # 与 R 程序一致：R 值小于 -1 时截断为 -1
    return (
        R,
        abnorm_x_list,
        abnorm_y_list,
        alam_days_arr,
        threshold_series,
        all_angle,
        error,
        all_angle_max_out,
        mean_value,
        std_value,
    )


def _distance_km(lon_a: float, lat_a: float, lon_b: float, lat_b: float) -> float:
    rad = np.pi / 180.0
    la, lb = lat_a * rad, lat_b * rad
    dlon = (lon_a - lon_b) * rad
    c = np.sin(la) * np.sin(lb) + np.cos(la) * np.cos(lb) * np.cos(dlon)
    c = np.clip(c, -1.0, 1.0)
    return float(6371.137 * np.arccos(c))


# ===========================
# 五、绘图与结果输出
# ===========================


def load_interference_periods(path: Optional[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取干扰信息文件，返回干扰起止时间（儒略日）。
    首行为标题，其后两列为开始/结束时间码；
    时间码解析失败的行跳过，仅返回有效起止对；支持 path 为 str 或 Path。
    """
    if not path:
        return np.array([], dtype=float), np.array([], dtype=float)
    path = Path(path)
    if not path.exists():
        return np.array([], dtype=float), np.array([], dtype=float)
    try:
        if path.stat().st_size == 0:
            return np.array([], dtype=float), np.array([], dtype=float)
    except OSError:
        return np.array([], dtype=float), np.array([], dtype=float)
    try:
        with open(path, encoding="utf-8") as f:
            arr = np.loadtxt(f, skiprows=1)
    except UnicodeDecodeError:
        with open(path, encoding="gbk") as f:
            arr = np.loadtxt(f, skiprows=1)
    except Exception:
        return np.array([], dtype=float), np.array([], dtype=float)
    if arr.ndim == 1:
        if arr.size < 2:
            return np.array([], dtype=float), np.array([], dtype=float)
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    start_codes = arr[:, 0]
    end_codes = arr[:, 1]
    if start_codes.size == 0 or end_codes.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    n = min(len(start_codes), len(end_codes))
    start_codes = start_codes[:n]
    end_codes = end_codes[:n]

    def _to_timestamp(c):
        try:
            return _parse_time_code_to_timestamp(int(float(c)))
        except (ValueError, TypeError):
            return None

    ts_start = [_to_timestamp(c) for c in start_codes]
    ts_end = [_to_timestamp(c) for c in end_codes]
    valid = [s is not None and e is not None for s, e in zip(ts_start, ts_end)]
    if not any(valid):
        return np.array([], dtype=float), np.array([], dtype=float)
    ts_start = [ts_start[i] for i in range(len(valid)) if valid[i]]
    ts_end = [ts_end[i] for i in range(len(valid)) if valid[i]]
    start_jd = _datetime_like_to_julian(ts_start)
    end_jd = _datetime_like_to_julian(ts_end)
    return start_jd.astype(float), end_jd.astype(float)


def plot_result_with_earthquakes(
    time_date: np.ndarray,
    pre_observed: np.ndarray,
    all_angle_max: np.ndarray,
    threshold_series: np.ndarray,
    alam_days_arr: np.ndarray,
    R: np.ndarray,
    angle_index: int,
    alarm_day_index: int,
    abnorm_x: np.ndarray,
    abnorm_y: np.ndarray,
    eq_dt_arr: np.ndarray,
    eq_lons: np.ndarray,
    eq_lats: np.ndarray,
    flag: np.ndarray,
    poly_x: np.ndarray,
    poly_y: np.ndarray,
    station_lon: float,
    station_lat: float,
    Rmax: float,
    R0: float,
    mag_start: float,
    out_png: Path,
    use_cartopy_map: bool = False,
    windows_length: int = 90,
) -> None:
    dt = _to_datetime_from_julian(time_date)
    eq_dt = _to_datetime_from_julian(eq_dt_arr)
    thresh = threshold_series[angle_index]
    alarm_day = int(alam_days_arr[alarm_day_index])
    dt_start, dt_end = dt.min(), dt.max()

    fig = plt.figure(figsize=(11, 7), dpi=120)
    fig.suptitle("Trend-related anomaly", fontsize=14)

    # -------- 子图 1：预处理曲线 + 异常点 + 角度拟合范围框 --------
    ax1 = fig.add_subplot(2, 2, 1)
    # 以各异常点为中心、左右各 windows_length 天为范围，框内纵轴为该段数据最小/最大值的矩形（深青虚线框，不填充）
    from matplotlib.patches import Rectangle
    from matplotlib.dates import date2num
    dark_cyan = (0.0, 0.45, 0.55)
    for ab in abnorm_x:
        x_left = float(ab) - windows_length
        x_right = float(ab) + windows_length
        mask = (time_date >= x_left) & (time_date <= x_right)
        if not np.any(mask):
            continue
        y_lo = float(np.min(pre_observed[mask]))
        y_hi = float(np.max(pre_observed[mask]))
        x_left_dt = _to_datetime_from_julian(np.array([x_left]))[0]
        x_right_dt = _to_datetime_from_julian(np.array([x_right]))[0]
        x_lo = date2num(x_left_dt)
        x_hi = date2num(x_right_dt)
        rect = Rectangle(
            (x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
            facecolor="none", edgecolor=dark_cyan, linestyle="--", transform=ax1.transData,
        )
        ax1.add_patch(rect)
    ax1.plot(dt, pre_observed, "k-", linewidth=1.2, label="Raw data")
    if len(abnorm_x) > 0:
        ax1.plot(
            _to_datetime_from_julian(abnorm_x),
            abnorm_y,
            "r.",
            markersize=10,
            label="Anomaly points",
        )
    ax1.set_xlabel("Time / Year")
    ax1.set_ylabel("Observation")
    ax1.set_xlim(dt_start, dt_end)
    from matplotlib.patches import Patch
    hand, lab = ax1.get_legend_handles_labels()
    hand.append(Patch(facecolor="none", edgecolor=dark_cyan, linestyle="--"))
    lab.append("Angle fitting range")
    ax1.legend(handles=hand, labels=lab, loc="best", fontsize=8)
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # -------- 子图 2：空间分布（多边形、台站、报准/漏报震例，可选 cartopy 底图） --------
    lon_min = float(np.min(poly_x)) - 0.5
    lon_max = float(np.max(poly_x)) + 0.5
    lat_min = float(np.min(poly_y)) - 0.5
    lat_max = float(np.max(poly_y)) + 0.5
    use_cartopy = False
    transform = None
    if use_cartopy_map:
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            ax2 = fig.add_subplot(2, 2, 2, projection=ccrs.PlateCarree())
            ax2.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            ax2.add_feature(cfeature.LAND, facecolor=(0.95, 0.95, 0.9))
            ax2.add_feature(cfeature.OCEAN, facecolor=(0.88, 0.94, 1.0))
            ax2.add_feature(cfeature.COASTLINE, linewidth=0.6)
            ax2.add_feature(cfeature.BORDERS, linestyle="-", linewidth=0.3)
            transform = ccrs.PlateCarree()
            use_cartopy = True
        except ImportError:
            ax2 = fig.add_subplot(2, 2, 2)
    else:
        ax2 = fig.add_subplot(2, 2, 2)

    def _plot_ax2(x, y, *args, **kwargs):
        if use_cartopy and transform is not None:
            ax2.plot(x, y, *args, transform=transform, **kwargs)
        else:
            ax2.plot(x, y, *args, **kwargs)

    _plot_ax2(poly_x, poly_y, "k-", linewidth=1.0, label="Polygon")
    _plot_ax2([station_lon], [station_lat], "p", markersize=10, markerfacecolor="y", markeredgecolor="k", label="Station")
    pred_idx = np.where(flag)[0]
    miss_idx = np.where(~flag)[0]
    if len(pred_idx) > 0:
        _plot_ax2(eq_lons[pred_idx], eq_lats[pred_idx], "ro", markersize=6)
    if len(miss_idx) > 0:
        _plot_ax2(eq_lons[miss_idx], eq_lats[miss_idx], "bo", markersize=6)
    from matplotlib.lines import Line2D
    h2, l2 = ax2.get_legend_handles_labels()
    h2 = [hh for hh, ll in zip(h2, l2) if ll not in ("Predicted EQ", "Unpredicted EQ")]
    h2.extend([
        Line2D([0], [0], marker="o", color="w", markerfacecolor="r", markersize=6, label="Predicted EQ"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="b", markersize=6, label="Unpredicted EQ"),
    ])
    ax2.legend(handles=h2, loc="best", fontsize=8)
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    if not use_cartopy:
        ax2.set_xlim(lon_min, lon_max)
        ax2.set_ylim(lat_min, lat_max)
    ax2.grid(True, linestyle=":", alpha=0.5)

    # -------- 子图 3：角度极值曲线 + 阈值线 + 异常区填充 + 预报窗 + 震例（风格与 R 程序一致） --------
    ax3 = fig.add_subplot(2, 2, 3)
    dark_purple = (0.25, 0, 0.45)
    ax3.plot(dt, all_angle_max, "-", color=dark_purple, linewidth=1, label="Extremum of deflection angle")
    ax3.axhline(thresh, color="r", linestyle="--", linewidth=1.2, label=f"Threshold={thresh:.0f} deg")
    data_min = float(np.nanmin(all_angle_max))
    data_max = float(np.nanmax(all_angle_max))
    data_range = data_max - data_min
    if data_range <= 0:
        data_range = 1.0
    y_min = data_min
    y_max = data_min + data_range * 1.1
    eq_line_top = data_min + data_range * 1.05
    ax3.set_ylim(y_min, y_max)
    # 异常区填充（本程序为单点异常，对每个异常点做小段填充）
    for ab in abnorm_x:
        ab_f = float(ab)
        mask = (time_date >= ab_f - 0.5) & (time_date <= ab_f + 0.5)
        if not np.any(mask):
            continue
        ax3.fill_between(
            np.array(dt)[mask], all_angle_max[mask], thresh,
            color=(250 / 255, 200 / 255, 205 / 255), alpha=0.8,
        )
    dark_green = (0.0, 0.4, 0.0)
    for ab in abnorm_x:
        pred_st = float(ab)
        pred_ed = pred_st + alarm_day
        st_dt = _to_datetime_from_julian(np.array([pred_st]))[0]
        ed_dt = _to_datetime_from_julian(np.array([pred_ed]))[0]
        ax3.plot([st_dt, ed_dt], [thresh, thresh], "-", color=dark_green, linewidth=4)
    for i in range(len(eq_dt_arr)):
        color = "r" if flag[i] else "b"
        ax3.plot([eq_dt[i], eq_dt[i]], [thresh, eq_line_top], "k-", linewidth=0.8)
        ax3.plot(eq_dt[i], eq_line_top, "o", color=color, markerfacecolor=color, markersize=6)
    ax3.set_xlim(dt_start, dt_end)
    ax3.set_xlabel(f"Threshold>={thresh:.0f} deg  Mag>={mag_start:.1f}")
    ax3.set_ylabel("Angle / deg")
    ax3.set_title(f"R={Rmax:.4f}, R0={R0:.4f}, alarm={alarm_day} days", fontsize=10, color="r")
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    h3, l3 = ax3.get_legend_handles_labels()
    h3 = [hh for hh, ll in zip(h3, l3) if ll not in ("Predicted EQ", "Unpredicted EQ")]
    h3.extend([
        Line2D([0], [0], marker="o", color="w", markerfacecolor="r", markersize=6, label="Predicted EQ"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="b", markersize=6, label="Unpredicted EQ"),
    ])
    ax3.legend(handles=h3, loc="best", fontsize=8)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax3.grid(True, linestyle=":", alpha=0.5)

    # -------- 子图 4：R-TT 网格（预测期 × 阈值上的 R 值，最优点标黑） --------
    ax4 = fig.add_subplot(2, 2, 4)
    thresh_2d = np.asarray(threshold_series)
    if thresh_2d.ndim == 0:
        thresh_2d = np.array([thresh_2d])
    # x = prediction time (days), y = threshold (deg)
    X, Y = np.meshgrid(alam_days_arr, thresh_2d)
    im = ax4.pcolormesh(X, Y, R, shading="auto", cmap="jet", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax4, label="R value")
    ax4.plot(alam_days_arr[alarm_day_index], threshold_series[angle_index], "ko", markerfacecolor="k", markersize=10)
    ax4.set_xlabel("Prediction time / days")
    ax4.set_ylabel("Threshold / deg")
    ax4.set_title("R-TT (R value vs Time & Threshold)")

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ===========================
# 六、主流程（单测项）
# ===========================


def run_single_station(
    data_file: Path,
    polygon_file: Path,
    eq_catalog_file: Path,
    station_lon: float,
    station_lat: float,
    interference_file: Optional[Path] = None,
    mag_min: float = 3.5,
    mag_max: float = 9.0,
    windows_length: int = 90,
    windows_step: int = 1,
    day_e: int = 720,
    day_s: int = 1,
    day_n: int = 1,
    thres_s: int = 50,
    thres_n: int = 1,
    thres_e: int = 180,
    use_cartopy_map: bool = False,
) -> None:
    """
    单测项趋势转折分析主流程。
    支持：2 列日值（8 位 yyyymmdd）、2 列整点值（10 位 yyyymmddhh）、多列（日期+24 小时值取日均）。
    仅在有震例时进行 R 值计算并输出图件与结果文件。
    """
    data_file = Path(data_file)
    polygon_file = Path(polygon_file)
    eq_catalog_file = Path(eq_catalog_file)

    if not data_file.exists():
        print(f"错误：观测数据文件不存在，无法继续。路径: {data_file}")
        return
    if not polygon_file.exists():
        print(f"错误：震例对应规则文件（多边形）不存在，无法继续。路径: {polygon_file}")
        return
    if not eq_catalog_file.exists():
        print(f"错误：地震目录文件不存在，无法继续。路径: {eq_catalog_file}")
        return

    out_dir = data_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"数据文件: {data_file.resolve()}")
    print(f"规则文件: {polygon_file.resolve()}")
    print(f"地震目录: {eq_catalog_file.resolve()}")
    time_date, obs_raw = read_data(data_file)
    if len(time_date) == 0 or len(obs_raw) == 0:
        print("错误：观测数据文件（DATA_FILE）内容为空或读取失败，无法继续，程序结束。")
        return

    # 判断整点值（10 位 yyyymmddhh）还是日值（8 位或已为 datenum）
    is_hourly = np.issubdtype(time_date.dtype, np.integer) and np.all(time_date >= 1e9)
    if is_hourly:
        mi_P = np.array([], dtype=np.int64)
        obs_pre = preprocess_missing_and_steps_hourly(obs_raw)
        days, day_vals = daily_mean_from_hourly(
            time_date.astype(np.int64), obs_pre
        )
        if len(days) == 0 or len(day_vals) == 0:
            print("错误：整点值转日均后无有效数据，无法继续，程序结束。")
            return
        time_date = _datetime_like_to_julian(
            pd.to_datetime(days.astype(str), format="%Y%m%d")
        )
        obs_pre = day_vals
        print("数据预处理完成（整点值→日均值，缺数与台阶已处理）")
    else:
        time_date = np.asarray(time_date, dtype=float)
        obs_pre, mi_P = data_preprocess(obs_raw)
        print("数据预处理完成（日值，缺数与台阶已处理）")

    st_t: np.ndarray = np.array([], dtype=float)
    ed_t: np.ndarray = np.array([], dtype=float)
    if interference_file and Path(interference_file).exists():
        st_t, ed_t = load_interference_periods(interference_file)
    # windows_length<90 时：将缺数位置并入干扰时段，在异常提取时一并剔除
    if not is_hourly and windows_length < 90 and len(mi_P) > 0:
        miss_times = time_date[mi_P].astype(float)
        st_t = np.concatenate([st_t, miss_times]) if len(st_t) > 0 else miss_times
        ed_t = np.concatenate([ed_t, miss_times]) if len(ed_t) > 0 else miss_times

    try:
        with open(polygon_file, encoding="utf-8") as f:
            poly = np.loadtxt(f, skiprows=1)
    except UnicodeDecodeError:
        try:
            with open(polygon_file, encoding="gbk") as f:
                poly = np.loadtxt(f, skiprows=1)
        except Exception as e:
            print(f"错误：规则文件（POLYGON_FILE）读取失败或编码不支持，无法继续，程序结束。{e}")
            return
    except Exception as e:
        print(f"错误：规则文件（POLYGON_FILE）读取失败或内容格式有误，无法继续，程序结束。{e}")
        return
    if poly.size == 0:
        print("错误：规则文件（POLYGON_FILE）内容为空（无有效顶点），无法继续，程序结束。")
        return
    if poly.ndim == 1:
        poly = poly.reshape(1, -1)
    poly_x = poly[:, 0]
    poly_y = poly[:, 1]
    if len(poly_x) == 0:
        print("错误：规则文件（POLYGON_FILE）内容为空（无有效顶点），无法继续，程序结束。")
        return

    cat = EarthquakeCatalog(eq_catalog_file)
    df_all = cat.load()
    if df_all.empty:
        print("错误：地震目录文件（EQ_CATALOG_FILE）内容为空，无法继续，程序结束。")
        return
    t_start = _from_matlab_datenum(float(time_date[0]))
    t_end = _from_matlab_datenum(float(time_date[-1]))
    df_sel = cat.select(
        start_time=t_start,
        end_time=t_end,
        mag_min=mag_min,
        mag_max=mag_max,
        polygon_lon=poly_x,
        polygon_lat=poly_y,
    )

    if len(df_sel) > 0:
        eq_times = _datetime_like_to_julian(df_sel["time"].tolist())
        eq_lons = df_sel["lon"].to_numpy()
        eq_lats = df_sel["lat"].to_numpy()
        eq_mags = df_sel["mag"].to_numpy()
        print(f"筛选到 {len(eq_times)} 个震例。")
        if len(st_t) > 0:
            print(f"成功读取 {len(st_t)} 段干扰时段，将在异常提取和 R 值计算中剔除这些时段。")
        R, abnorm_x_list, abnorm_y_list, alam_days_arr, threshold_series, all_angle, error, all_angle_max, mean_value, std_value = rtt(
            time_date,
            obs_pre,
            eq_times,
            st_t,
            ed_t,
            day_e=day_e,
            windows_length=windows_length,
            windows_step=windows_step,
            thres_s=thres_s,
            thres_n=thres_n,
            thres_e=thres_e,
            day_s=day_s,
            day_n=day_n,
        )
        Rmax = float(np.max(R))
        idx = np.unravel_index(np.argmax(R), R.shape)
        angle_index = int(idx[0])
        alarm_day_index = int(idx[1])
        abnorm_x_best = abnorm_x_list[angle_index]
        abnorm_y_best = abnorm_y_list[angle_index]
        best_alarm_day = int(alam_days_arr[alarm_day_index])
        treating = data_file.stem
        print("角度值已完成拟合和极值提取")
        # 输出 processed 文件：时间、筛选并剔除干扰后的角度极值、是否异常(1/0)
        out_processed = out_dir / f"{treating}_processed.txt"
        with open(out_processed, "w", encoding="utf-8") as f:
            f.write("Time\tFitted_Angle\tIs_Anomaly\n")
            for i in range(len(time_date)):
                t = float(time_date[i])
                time_str = _from_matlab_datenum(t).strftime("%Y%m%d")
                angle_val = float(all_angle_max[i])
                is_anomaly = 1 if np.any(np.abs(np.asarray(abnorm_x_best) - t) < 1e-6) else 0
                f.write(f"{time_str}\t{angle_val:.3f}\t{is_anomaly}\n")
        print(f"处理序列已写入: {out_processed.resolve()}")
        if len(abnorm_x_best) > 0:
            # 预报窗结束与 R 程序一致：每段（本程序为单点）窗长 = max(段长, 预报期)，此处段长=1
            ab_ed_best = np.array(
                [float(st) + max(1, best_alarm_day) for st in abnorm_x_best],
                dtype=float,
            )
            flags = is_predicted(eq_times, np.asarray(abnorm_x_best, dtype=float), ab_ed_best, best_alarm_day)
            success = int(np.sum(flags))
            miss = len(eq_times) - success
            R0 = get_r0(success, miss)
            out_txt = out_dir / f"{treating}_result.txt"
            jy = "通过检验（R > R0）" if (Rmax > R0 and Rmax > 0) else "未通过检验（R < R0）"
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write("# 趋势转折（速率变化）异常提取及预报效能结果（单测项）\n")
                f.write(f"# 观测数据文件: {data_file.name}\n")
                f.write(f"# 地震目录文件: {eq_catalog_file.name}\n")
                f.write(f"# 地震筛选规则文件: {polygon_file.name}\n")
                if interference_file is not None:
                    f.write(f"# 干扰信息文件: {Path(interference_file).name}\n")
                else:
                    f.write("# 干扰信息文件: 无\n")
                f.write(f"# 台站经纬度: {station_lon:.3f}, {station_lat:.3f}\n")
                f.write("#---------------------------------------------\n")
                f.write(f"#------------------------------------ {treating} ({jy}) -----------------------------------\n")
                f.write("# 异常次数   成功预报地震数   漏报地震数    R    R0   最优预报有效期   最优阈值(°)  震级下限  震级上限\n")
                f.write(f"{len(abnorm_x_best):6d} {success:16d} {miss:16d} {Rmax:8.4f} {R0:8.4f} {best_alarm_day:4d} {threshold_series[angle_index]:18.0f} {mag_min:8.1f} {mag_max:8.1f}\n")
                f.write("# 具体如下：\n")
                f.write("# 异常开始时间    异常最后时间    是否报准地震   异常距地震发生时间(天)  地震发生时间  地震经度    地震纬度    地震震级    震中距(Km)\n")
                # 异常最后时间 = 段终点（本程序为单点异常，段终点=段起点）；报准判定用预报窗 win_ed
                eq_floor = np.floor(eq_times).astype(int)
                for i in range(len(abnorm_x_best)):
                    st = float(abnorm_x_best[i])
                    ed = st  # 单点异常，段终点 = 段起点
                    win_ed = st + best_alarm_day  # 预报窗结束，仅用于报准判定
                    st_int = int(np.ceil(st))
                    ed_int = int(np.floor(win_ed))
                    eq_in = (eq_floor >= st_int) & (eq_floor <= ed_int)
                    st_str = _from_matlab_datenum(st).strftime("%Y%m%d")
                    ed_str = _from_matlab_datenum(ed).strftime("%Y%m%d")  # 异常最后时间 = 段终点
                    if not np.any(eq_in):
                        f.write(f"{st_str}  {ed_str}      否\n")
                    else:
                        for j in np.where(eq_in)[0]:
                            days_st = int(np.floor(eq_times[j] - st))
                            days_ed = int(np.floor(eq_times[j] - ed))
                            eq_date_str = _from_matlab_datenum(eq_times[j]).strftime("%Y%m%d")
                            dist_km = _distance_km(eq_lons[j], eq_lats[j], station_lon, station_lat)
                            f.write(f"{st_str}  {ed_str}      是    {days_st:d}/{days_ed:d}  {eq_date_str}  {eq_lons[j]:14.2f}  {eq_lats[j]:5.2f}  {eq_mags[j]:5.1f}  {dist_km:5.1f}\n")
            png_path = out_dir / f"{treating}_result.png"
            try:
                plot_result_with_earthquakes(
                    time_date, obs_pre, all_angle_max, threshold_series, alam_days_arr, R,
                    angle_index, alarm_day_index, abnorm_x_best, abnorm_y_best,
                    eq_times, eq_lons, eq_lats, flags, poly_x, poly_y,
                    station_lon, station_lat, Rmax, R0, mag_min, png_path,
                    use_cartopy_map=use_cartopy_map,
                    windows_length=windows_length,
                )
                print(f"图件已保存: {png_path.resolve()}")
            except Exception as e:
                print(f"绘图异常（不影响数值结果）: {e}")
            print("计算完成，关键结果如下：")
            print(f"  Rmax = {Rmax:.4f}")
            print(f"  R0   = {R0:.4f}")
            print(f"  最优阈值 = {threshold_series[angle_index]:.0f}°")
            print(f"  最优预报期 = {best_alarm_day}天")
            print(f"  成功报准 / 总地震数 = {success} / {len(eq_times)}")
        else:
            print("无满足要求的异常点。")
    else:
        print("无震例，程序结束（本程序仅在有震例时输出结果与图件）。")


if __name__ == "__main__":
    # ========= 请按实际情况修改以下参数 =========
    DATA_FILE = Path(r"D:\numerical\trend-related-anomaly\14001_1_2231.txt")  # 测项日均值观测数据文件（2列：日期、值；或25列：日期+24小时值）
    POLYGON_FILE = Path(r"D:\numerical\trend-related-anomaly\14001_1_2231_EQ_SelectRule.txt")  # 震例对应规则（多边形经纬度）
    EQ_CATALOG_FILE = Path(r"D:\numerical\trend-related-anomaly\china3ms.eqt")  # 地震目录 EQT 格式
    INTERFERENCE_FILE = Path(r"D:\numerical\trend-related-anomaly\14001_1_2231_Interference_Period.txt") # 干扰信息表路径（可选），无则 None
    STATION_LON = 112.434 # 台站经度
    STATION_LAT = 37.713 # 台站纬度
    MAG_MIN = 4.0 # 震级下限（地震目录筛选）
    MAG_MAX = 9.0 # 震级上限
    WINDOWS_LENGTH = 180  # 滑动窗口长度（天）
    WINDOWS_STEP = 1      # 滑动步长（天）
    # 预报期序列由主函数赋值（DAY_S, DAY_N, DAY_E）
    DAY_S = 1   # 预报有效期下限（天）
    DAY_N = 1   # 预报有效期步长（天）
    DAY_E = 360 # 预报有效期上限（天）
    THRES_S, THRES_N, THRES_E = 50, 1, 180 # 阈值扫描范围（度）
    USE_CARTOPY_MAP = False  # 是否在右上角空间分布图添加 cartopy 底图

    if not DATA_FILE.exists():
        print(f"请将 DATA_FILE 改为实际观测数据路径。当前: {DATA_FILE}")
        exit(1)
    if not POLYGON_FILE.exists():
        print(f"请将 POLYGON_FILE 改为实际多边形规则路径。当前: {POLYGON_FILE}")
        exit(1)
    if not EQ_CATALOG_FILE.exists():
        print(f"请将 EQ_CATALOG_FILE 改为实际地震目录路径。当前: {EQ_CATALOG_FILE}")
        exit(1)

    run_single_station(
        data_file=DATA_FILE,
        polygon_file=POLYGON_FILE,
        eq_catalog_file=EQ_CATALOG_FILE,
        station_lon=STATION_LON,
        station_lat=STATION_LAT,
        interference_file=INTERFERENCE_FILE,
        mag_min=MAG_MIN,
        mag_max=MAG_MAX,
        windows_length=WINDOWS_LENGTH,
        windows_step=WINDOWS_STEP,
        day_s=DAY_S,
        day_n=DAY_N,
        day_e=DAY_E,
        thres_s=THRES_S,
        thres_n=THRES_N,
        thres_e=THRES_E,
        use_cartopy_map=USE_CARTOPY_MAP,
    )
