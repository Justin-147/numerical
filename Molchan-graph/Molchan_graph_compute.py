# -*- coding: utf-8 -*-
"""
Molchan-graph 计算与成图：读取已含时间列与特征列的数据文件(通常该数据来自异常提取方法的处理输出结果)，以及地震目录、规则文件、干扰信息，
在阈值×预测期网格上计算R值，输出结果文件与图件。
图件布局：左上=空间分布，右上=R-TT二维，左下=残差+阈值+预警+震例，右下=最优预测期下随阈值变化的Molchan图
"""

import warnings

from pathlib import Path
from typing import List, Tuple

warnings.filterwarnings("ignore", category=UserWarning, message=".*[Dd]iscarding nonzero nanoseconds.*")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ===========================
# 一、地震目录与工具
# ===========================

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


def _parse_time_code_to_timestamp(code: int) -> pd.Timestamp:
    s = str(int(code))
    if len(s) == 14:
        y, mo, d = int(s[0:4]), int(s[4:6]), int(s[6:8])
        h, mi, sec = int(s[8:10]), int(s[10:12]), int(s[12:14])
    elif len(s) == 12:
        y, mo, d = int(s[0:4]), int(s[4:6]), int(s[6:8])
        h, mi, sec = int(s[8:10]), int(s[10:12]), 0
    elif len(s) == 10:
        y, mo, d = int(s[0:4]), int(s[4:6]), int(s[6:8])
        h, mi, sec = int(s[8:10]), 0, 0
    elif len(s) == 8:
        y, mo, d = int(s[0:4]), int(s[4:6]), int(s[6:8])
        h, mi, sec = 0, 0, 0
    elif len(s) == 6:
        y, mo, d = int(s[0:4]), int(s[4:6]), 1
        h, mi, sec = 0, 0, 0
    elif len(s) == 4:
        y, mo, d = int(s[0:4]), 1, 1
        h, mi, sec = 0, 0, 0
    else:
        return pd.to_datetime(s)
    return pd.Timestamp(year=y, month=mo, day=d, hour=h, minute=mi, second=sec)


def _to_datetime_from_julian(times: np.ndarray) -> np.ndarray:
    return np.array([_from_matlab_datenum(t).to_pydatetime() for t in np.atleast_1d(times).ravel()])


def _distance_km(lon_a: float, lat_a: float, lon_b: float, lat_b: float) -> float:
    rad = np.pi / 180.0
    la, lb = lat_a * rad, lat_b * rad
    dlon = (lon_a - lon_b) * rad
    c = np.sin(la) * np.sin(lb) + np.cos(la) * np.cos(lb) * np.cos(dlon)
    c = np.clip(c, -1.0, 1.0)
    return float(6371.137 * np.arccos(c))


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
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df
        lines = self.file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        records = []
        for line in lines:
            if not line.strip():
                continue
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
            records.append({
                "time": pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute, second=second),
                "lon": lon, "lat": lat, "mag": mag, "depth": depth,
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
# 二、数据与规则文件读取
# ===========================


def _split_line(line: str, delim: str | None) -> List[str]:
    if delim is None:
        return line.split()
    return line.split(delim)


def read_data_file(
    path: Path,
    has_header: bool = True,
    delimiter: str = "\t",
    encoding: str = "utf-8",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取两列数据文件：第 1 列时间，第 2 列数值。支持有/无表头、中文或英文表头。
    时间支持：日值 8 位（yyyymmdd）、整点值 10 位（yyyymmddhh）、分钟值 12 位（yyyymmddhhmm），
    统一转换为儒略日（浮点，含日内小数）。
    返回 (times_julian, values)。
    """
    try:
        with open(path, encoding=encoding) as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(path, encoding="gbk") as f:
            lines = f.readlines()
    if not lines:
        return np.array([]), np.array([])
    start = 1 if has_header else 0
    data_lines = [ln.rstrip("\n\r") for ln in lines[start:] if ln.strip()]
    times_code = []
    residual = []
    for line in data_lines:
        cols = _split_line(line, delimiter if delimiter else "\t")
        if len(cols) < 2:
            continue
        try:
            tc = int(float(cols[0]))
            val = float(cols[1])
        except (ValueError, TypeError):
            continue
        times_code.append(tc)
        residual.append(val)
    if not times_code:
        return np.array([]), np.array([])
    times_code = np.array(times_code, dtype=np.int64)
    residual = np.array(residual, dtype=float)
    times_julian = np.array(
        [_to_matlab_datenum(_parse_time_code_to_timestamp(int(t))) for t in times_code],
        dtype=float,
    )
    return times_julian, residual


def read_polygon(path: Path, encoding: str = "utf-8") -> Tuple[np.ndarray, np.ndarray]:
    try:
        with open(path, encoding=encoding) as f:
            poly = np.loadtxt(f, skiprows=1)
    except UnicodeDecodeError:
        with open(path, encoding="gbk") as f:
            poly = np.loadtxt(f, skiprows=1)
    if poly.ndim == 1:
        poly = poly.reshape(1, -1)
    return poly[:, 0], poly[:, 1]


def load_interference_periods(path: Path | None) -> Tuple[np.ndarray, np.ndarray]:
    if not path or not path.exists():
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
    n = min(len(start_codes), len(end_codes))
    start_codes = start_codes[:n]
    end_codes = end_codes[:n]

    def to_ts(c):
        try:
            return _parse_time_code_to_timestamp(int(float(c)))
        except (ValueError, TypeError):
            return None

    ts_start = [to_ts(c) for c in start_codes]
    ts_end = [to_ts(c) for c in end_codes]
    valid = [s is not None and e is not None for s, e in zip(ts_start, ts_end)]
    if not any(valid):
        return np.array([], dtype=float), np.array([], dtype=float)
    ts_start = [ts_start[i] for i in range(len(valid)) if valid[i]]
    ts_end = [ts_end[i] for i in range(len(valid)) if valid[i]]
    start_jd = _datetime_like_to_julian(ts_start)
    end_jd = _datetime_like_to_julian(ts_end)
    return start_jd.astype(float), end_jd.astype(float)


# ===========================
# 三、异常检测与 R 值
# ===========================


def detect_anomalies_threshold(
    times: np.ndarray,
    residual: np.ndarray,
    std_pd: float,
    rate: float,
    interf_start: np.ndarray | None = None,
    interf_end: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    abs_err = np.abs(residual)
    n = len(residual)
    idx_st: List[int] = []
    idx_ed: List[int] = []
    pnt_st = False
    for i in range(n):
        if abs_err[i] > rate * std_pd:
            if not pnt_st:
                idx_st.append(i)
                pnt_st = True
        elif pnt_st:
            idx_ed.append(i - 1)  # 段终点 = 最后仍超阈值的下标（i 为首次低于阈值的下标）
            pnt_st = False
        if i == n - 1 and pnt_st:
            idx_ed.append(n - 1)
    if not idx_st:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )
    idx_st_arr = np.asarray(idx_st, dtype=int)
    idx_ed_arr = np.asarray(idx_ed, dtype=int)
    time_st = times[idx_st_arr]
    time_ed = times[idx_ed_arr]
    time_section = time_ed - time_st
    if interf_start is not None and interf_end is not None and len(interf_start) > 0:
        keep_mask = np.ones(len(time_st), dtype=bool)
        for j in range(len(interf_start)):
            st_j, ed_j = float(interf_start[j]), float(interf_end[j])
            overlap = ~((time_ed < st_j) | (time_st > ed_j))
            keep_mask &= ~overlap
        time_st = time_st[keep_mask]
        time_ed = time_ed[keep_mask]
        time_section = time_section[keep_mask]
        idx_st_arr = idx_st_arr[keep_mask]
        idx_ed_arr = idx_ed_arr[keep_mask]
    abn_ampl = np.array(
        [float(np.max(residual[idx_st_arr[i] : idx_ed_arr[i] + 1])) for i in range(len(idx_st_arr))],
        dtype=float,
    )
    return time_st, time_ed, time_section, abn_ampl


def detect_anomalies_absolute_threshold(
    times: np.ndarray,
    residual: np.ndarray,
    thresh: float,
    interf_start: np.ndarray | None = None,
    interf_end: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    按给定绝对值阈值提取异常段：|residual| > thresh 的连续区间。
    返回 time_st, time_ed, time_section, abn_ampl。time_ed 为段终点（最后仍超阈值日）。
    """
    abs_err = np.abs(residual)
    n = len(residual)
    idx_st: List[int] = []
    idx_ed: List[int] = []
    pnt_st = False
    for i in range(n):
        if abs_err[i] > thresh:
            if not pnt_st:
                idx_st.append(i)
                pnt_st = True
        elif pnt_st:
            idx_ed.append(i - 1)  # 段终点 = 最后仍超阈值的下标（i 为首次低于阈值的下标）
            pnt_st = False
        if i == n - 1 and pnt_st:
            idx_ed.append(n - 1)
    if not idx_st:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )
    idx_st_arr = np.asarray(idx_st, dtype=int)
    idx_ed_arr = np.asarray(idx_ed, dtype=int)
    time_st = times[idx_st_arr]
    time_ed = times[idx_ed_arr]
    time_section = time_ed - time_st
    if interf_start is not None and interf_end is not None and len(interf_start) > 0:
        keep_mask = np.ones(len(time_st), dtype=bool)
        for j in range(len(interf_start)):
            st_j, ed_j = float(interf_start[j]), float(interf_end[j])
            overlap = ~((time_ed < st_j) | (time_st > ed_j))
            keep_mask &= ~overlap
        time_st = time_st[keep_mask]
        time_ed = time_ed[keep_mask]
        time_section = time_section[keep_mask]
        idx_st_arr = idx_st_arr[keep_mask]
        idx_ed_arr = idx_ed_arr[keep_mask]
    abn_ampl = np.array(
        [float(np.max(residual[idx_st_arr[i] : idx_ed_arr[i] + 1])) for i in range(len(idx_st_arr))],
        dtype=float,
    )
    return time_st, time_ed, time_section, abn_ampl


def _del_itf_time(
    time_st: np.ndarray,
    time_ed: np.ndarray,
    itf_st: np.ndarray,
    itf_ed: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(time_st) == 0:
        return time_st, time_ed
    st = time_st.astype(float).copy()
    ed = time_ed.astype(float).copy()
    for i in range(len(st)):
        orig_st, orig_ed = float(st[i]), float(ed[i])
        for j in range(len(itf_st)):
            if orig_st <= itf_ed[j] and orig_ed >= itf_st[j]:
                if itf_st[j] < orig_st:
                    st[i] = np.nan
                    ed[i] = np.nan
                    break
                else:
                    ed[i] = float(itf_st[j])
    keep = ~(np.isnan(st) | np.isnan(ed))
    return st[keep], ed[keep]


def is_predicted(
    eq_times: np.ndarray,
    ab_st: np.ndarray,
    ab_ed: np.ndarray,
    alarm_day: int,
) -> np.ndarray:
    """
    报准判定：判断每个地震是否落在任一异常段的预报窗内。
    预报窗规则：对每段异常 [st, ed]，若段长 > 预报期则预报窗为 [st, ed]，否则为 [st, st+alarm_day]（取长者）。
    判定方式：地震时间取整到天 floor(eq)，若落在任意一段的预报窗 [st, ed] 内即视为报准。
    返回与 eq_times 同长的 bool 数组，True 表示该地震被报准。
    """
    pred_intervals: List[Tuple[float, float]] = []
    for st, ed in zip(ab_st, ab_ed):
        if ed - st > alarm_day:
            pred_intervals.append((float(st), float(ed)))
        else:
            pred_intervals.append((float(st), float(st + alarm_day)))
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
    R 值计算：报准率 - 时间占用率。
    报准率：报准地震数/地震总数
    时间占用率：异常段长度之和/总时间跨度
    预报窗结束 = 段开始 + max(段长, alarm_days)（取长者），再交 is_predicted 做报准判定。
    """
    n_eq = max(len(eq_times), 1)
    dt_span = max(float(dt_end - dt_start), 1e-6)
    if len(alarm_start) == 0:
        return 0.0, 0.0, 0.0
    # 每段预警结束 = 开始+持续 或 开始+预报期（取长者），与 is_predicted 中预报窗一致
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


def r_value_grid(
    times: np.ndarray,
    residual: np.ndarray,
    std_pd: float,
    eq_times: np.ndarray,
    rate_min: float,
    rate_max: float,
    rate_step: float,
    alm_day_start: int,
    alm_day_end: int,
    alm_day_step: int = 1,
    interf_start: np.ndarray | None = None,
    interf_end: np.ndarray | None = None,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
    rate_max_auto = int(np.fix(np.max(np.abs(residual)) / std_pd) + 1)
    rate_max = min(rate_max, rate_max_auto)
    rate_series = np.arange(rate_min, rate_max, rate_step)
    if len(rate_series) == 0:
        rate_series = np.array([rate_min], dtype=float)
    alm_days_arr = np.arange(alm_day_start, alm_day_end + 1, alm_day_step)
    R_all = np.zeros((len(rate_series), len(alm_days_arr)), dtype=float)
    abnorm_x_list: List[np.ndarray] = []
    abnorm_y_list: List[np.ndarray] = []
    dt_start = float(times[0])
    dt_end = float(times[-1])
    has_interf = (
        interf_start is not None and interf_end is not None and len(interf_start) > 0
    )
    for i, rate in enumerate(rate_series):
        time_st, time_ed, time_sec, _ = detect_anomalies_threshold(
            times, residual, std_pd, rate, interf_start=None, interf_end=None
        )
        if has_interf:
            abn_st, abn_ed = _del_itf_time(time_st, time_ed, interf_start, interf_end)
            abn_sec = abn_ed - abn_st
        else:
            abn_st, abn_ed, abn_sec = time_st, time_ed, time_sec
        if len(abn_st) == 0:
            break
        abnorm_x_list.append(abn_st)
        abnorm_y_list.append(abn_ed)
        for j, alm_day in enumerate(alm_days_arr):
            r_val, _, _ = r_value_single(eq_times, abn_st, abn_sec, alm_day, dt_start, dt_end)
            R_all[i, j] = r_val
    n_valid = len(abnorm_x_list)
    R_all = R_all[:n_valid, :]
    rate_series = rate_series[:n_valid]
    return R_all, abnorm_x_list, abnorm_y_list, alm_days_arr, rate_series


def r_value_grid_absolute(
    times: np.ndarray,
    residual: np.ndarray,
    eq_times: np.ndarray,
    thresh_start: float,
    thresh_end: float,
    thresh_step: float,
    alm_day_start: int,
    alm_day_end: int,
    alm_day_step: int = 1,
    interf_start: np.ndarray | None = None,
    interf_end: np.ndarray | None = None,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
    """
    按给定阈值范围（数据单位）生成阈值序列，在 (阈值, 预测期) 网格上计算 R 矩阵。
    返回 R, abnorm_x_list, abnorm_y_list, alm_days_arr, threshold_series。
    """
    threshold_series = np.arange(thresh_start, thresh_end + 1e-9, thresh_step, dtype=float)
    if len(threshold_series) == 0:
        threshold_series = np.array([thresh_start], dtype=float)
    alm_days_arr = np.arange(alm_day_start, alm_day_end + 1, alm_day_step)
    R_all = np.zeros((len(threshold_series), len(alm_days_arr)), dtype=float)
    abnorm_x_list: List[np.ndarray] = []
    abnorm_y_list: List[np.ndarray] = []
    dt_start = float(times[0])
    dt_end = float(times[-1])
    has_interf = (
        interf_start is not None and interf_end is not None and len(interf_start) > 0
    )
    for i, thresh in enumerate(threshold_series):
        time_st, time_ed, time_sec, _ = detect_anomalies_absolute_threshold(
            times, residual, thresh, interf_start=None, interf_end=None
        )
        if has_interf:
            abn_st, abn_ed = _del_itf_time(time_st, time_ed, interf_start, interf_end)
            abn_sec = abn_ed - abn_st
        else:
            abn_st, abn_ed, abn_sec = time_st, time_ed, time_sec
        abnorm_x_list.append(abn_st)
        abnorm_y_list.append(abn_ed)
        for j, alm_day in enumerate(alm_days_arr):
            r_val, _, _ = r_value_single(eq_times, abn_st, abn_sec, alm_day, dt_start, dt_end)
            R_all[i, j] = r_val
    return R_all, abnorm_x_list, abnorm_y_list, alm_days_arr, threshold_series


def _significance_level_binomial(N: int, h: int, tau: float) -> float:
    """二项分布下显著性水平 α = P(报准数 >= h | 占有率 τ)。"""
    if tau <= 0:
        return 1.0 if h == 0 else 0.0
    if tau >= 1:
        return 1.0
    from math import comb
    return sum(comb(N, k) * (tau**k) * ((1 - tau) ** (N - k)) for k in range(h, N + 1))


def _significance_level(N: int, h: int, tau: float, use_normal: bool = True) -> float:
    """
    显著性水平 α = P(报准数 >= h | 占有率 τ)。
    - (N==h & tao==1) | (h==0 & tao==0) → 1
    - N>=20 用正态近似：X=N/(N+1)*(h-N*tao)/sqrt(N*tao*(1-tao))，α=P(Z>X)=1-normcdf(X)
    - 否则二项：α=sum_{k=h..N} C(N,k)*tao^k*(1-tao)^(N-k)
    """
    if (N == h and tau == 1) or (h == 0 and tau == 0):
        return 1.0
    if tau <= 0:
        return 1.0 if h == 0 else 0.0
    if tau >= 1:
        return 1.0
    if N >= 20 and use_normal:
        from scipy import stats
        denom = np.sqrt(N * tau * (1 - tau))
        if denom < 1e-12:
            return 0.5
        X = (N / (N + 1)) * (h - N * tau) / denom
        if np.isinf(X):
            return 0.0 if X > 0 else 1.0
        return float(stats.norm.sf(X))
    return _significance_level_binomial(N, h, tau)


def _solve_tao_for_alpha(N: int, h: int, alpha: float = 0.025, tol: float = 1e-5) -> float:
    """
    求 τ 使得 _significance_level(N, h, τ) = alpha。
    二分法，初始 tao1=0、tao2=1，收敛判据 delta=1e-5；不单独处理 h=N，一律二分求解。
    """
    if h > N or h < 1:
        return np.nan
    use_normal = N >= 20
    low, high = 0.0, 1.0
    a1 = _significance_level(N, h, low, use_normal) - alpha
    a2 = _significance_level(N, h, high, use_normal) - alpha
    if abs(a1) <= tol:
        return low
    if abs(a2) <= tol:
        return high
    for _ in range(200):
        mid = (low + high) / 2
        a3 = _significance_level(N, h, mid, use_normal) - alpha
        if abs(a3) <= tol:
            return mid
        if a3 * a1 > 0 and a3 * a2 < 0:
            low = mid
            a1 = a3
        elif a3 * a1 < 0 and a3 * a2 > 0:
            high = mid
            a2 = a3
        else:
            return np.nan
    return (low + high) / 2


def _molchan_auc(tau_curve: np.ndarray, nu_curve: np.ndarray) -> float:
    """
    曲线左下方面积 AUC：
    XX=[0,0,Coverrate,0], YY=[0,1,1-Hits/N,0]，多边形 (0,0),(0,1),(τ,ν)...,(0,0)。
    """
    if len(tau_curve) == 0 or len(nu_curve) == 0:
        return 0.0
    tau = np.asarray(tau_curve).ravel()
    nu = np.asarray(nu_curve).ravel()
    x = np.concatenate([[0.0, 0.0], tau, [0.0]])
    y = np.concatenate([[0.0, 1.0], nu, [0.0]])
    return 0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def get_R0(success_count: int, miss_count: int) -> float:
    k = success_count
    n = success_count + miss_count
    if n == 0:
        return 0.0
    alpha = 0.025
    min_alpha = 1e38
    min_alpha_p = 0.0
    from math import comb
    for P in np.linspace(0.0, 1.0, 1001):
        s = sum(comb(n, i) * (P**i) * ((1 - P) ** (n - i)) for i in range(k, n + 1))
        diff = abs(s - alpha)
        if diff < min_alpha:
            min_alpha = diff
            min_alpha_p = P
    return k / n - min_alpha_p


# ===========================
# 四、成图（左上=地图，右上=R-TT二维，左下=残差+阈值+预警+震例，右下=最优预测期下随阈值变化的Molchan图）
# ===========================


def _molchan_curve_at_alarm_day(
    abnorm_x_list: List[np.ndarray],
    abnorm_y_list: List[np.ndarray],
    eq_times: np.ndarray,
    alarm_day: int,
    dt_start: float,
    dt_end: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在固定预报期下，按各阈值对应的异常段计算 (时间占有率 τ, 漏报率 ν)，
    用于绘制 Molchan 图。τ = 占有率，ν = 1 - 报准率。
    返回 (tau_arr, nu_arr) 已按 τ 升序排列，且首点为 (0, 1)。
    """
    n_eq = max(len(eq_times), 1)
    dt_span = max(float(dt_end - dt_start), 1e-6)
    tau_list: List[float] = []
    nu_list: List[float] = []
    for ab_st, ab_ed in zip(abnorm_x_list, abnorm_y_list):
        ab_sec = ab_ed - ab_st
        _, success_rate, occupied_rate = r_value_single(
            eq_times, ab_st, ab_sec, alarm_day, dt_start, dt_end
        )
        tau_list.append(occupied_rate)
        nu_list.append(1.0 - success_rate)
    if len(tau_list) == 0:
        return np.array([0.0, 1.0]), np.array([1.0, 0.0])
    tau_arr = np.array(tau_list, dtype=float)
    nu_arr = np.array(nu_list, dtype=float)
    order = np.argsort(tau_arr)
    tau_arr = tau_arr[order]
    nu_arr = nu_arr[order]
    tau_curve = np.concatenate([[0.0], tau_arr, [1.0]])
    nu_curve = np.concatenate([[1.0], nu_arr, [0.0]])
    return tau_curve, nu_curve


def plot_result_figure(
    times: np.ndarray,
    residual: np.ndarray,
    eq_times: np.ndarray,
    eq_lons: np.ndarray,
    eq_lats: np.ndarray,
    eq_flags: np.ndarray,
    poly_x: np.ndarray,
    poly_y: np.ndarray,
    station_lon: float,
    station_lat: float,
    threshold_series: np.ndarray,
    alam_days: np.ndarray,
    R: np.ndarray,
    best_threshold_idx: int,
    best_alarm_idx: int,
    ab_st_best: np.ndarray,
    ab_ed_best: np.ndarray,
    best_alarm_day: int,
    mag_min: float,
    Rmax: float,
    R0: float,
    out_png: Path,
    threshold_mode: int = 1,
    std_pd: float = 1.0,
    use_cartopy_map: bool = False,
    abnorm_x_list: List[np.ndarray] | None = None,
    abnorm_y_list: List[np.ndarray] | None = None,
) -> None:
    dt = _to_datetime_from_julian(times)
    eq_dt = _to_datetime_from_julian(eq_times)
    abs_err = np.abs(residual)

    fig = plt.figure(figsize=(11, 7), dpi=120)
    fig.suptitle("Molchan-graph result", fontsize=14)

    lon_min = float(np.min(poly_x)) - 0.5
    lon_max = float(np.max(poly_x)) + 0.5
    lat_min = float(np.min(poly_y)) - 0.5
    lat_max = float(np.max(poly_y)) + 0.5
    use_cartopy = False
    transform = None

    # -------- 左上：空间分布 --------
    if use_cartopy_map:
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
            ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            ax1.add_feature(cfeature.LAND, facecolor=(0.95, 0.95, 0.9))
            ax1.add_feature(cfeature.OCEAN, facecolor=(0.88, 0.94, 1.0))
            ax1.add_feature(cfeature.COASTLINE, linewidth=0.6)
            ax1.add_feature(cfeature.BORDERS, linestyle="-", linewidth=0.3)
            transform = ccrs.PlateCarree()
            use_cartopy = True
        except ImportError:
            ax1 = fig.add_subplot(2, 2, 1)
    else:
        ax1 = fig.add_subplot(2, 2, 1)

    def _plot_ax1(x, y, *args, **kwargs):
        if use_cartopy and transform is not None:
            ax1.plot(x, y, *args, transform=transform, **kwargs)
        else:
            ax1.plot(x, y, *args, **kwargs)

    _plot_ax1(poly_x, poly_y, "k-", linewidth=1.0, label="Polygon")
    _plot_ax1([station_lon], [station_lat], "p", markersize=10, markerfacecolor="y", markeredgecolor="k", label="Station")
    pred_idx = np.where(eq_flags)[0]
    miss_idx = np.where(~eq_flags)[0]
    if pred_idx.size > 0:
        _plot_ax1(eq_lons[pred_idx], eq_lats[pred_idx], "ro", markersize=6)
    if miss_idx.size > 0:
        _plot_ax1(eq_lons[miss_idx], eq_lats[miss_idx], "bo", markersize=6)
    from matplotlib.lines import Line2D
    h1, l1 = ax1.get_legend_handles_labels()
    h1 = [h for h, ll in zip(h1, l1) if ll not in ("Predicted EQ", "Unpredicted EQ")]
    h1.extend([
        Line2D([0], [0], marker="o", color="w", markerfacecolor="r", markersize=6, label="Predicted EQ"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="b", markersize=6, label="Unpredicted EQ"),
    ])
    ax1.legend(handles=h1, loc="best", fontsize=8)
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    if not use_cartopy:
        ax1.set_xlim(lon_min, lon_max)
        ax1.set_ylim(lat_min, lat_max)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # -------- 右上：R-TT 二维 --------
    ax2 = fig.add_subplot(2, 2, 2)
    X, Y = np.meshgrid(alam_days, threshold_series)
    c2 = ax2.pcolormesh(X, Y, R, shading="auto", cmap="jet", vmin=-1.0, vmax=1.0)
    fig.colorbar(c2, ax=ax2, label="R value")
    ax2.plot(alam_days[best_alarm_idx], threshold_series[best_threshold_idx], "ko", markersize=8, markerfacecolor="k")
    ax2.set_xlabel("Prediction time / days")
    ax2.set_ylabel("Threshold (×STD)" if threshold_mode == 1 else "Threshold")
    ax2.set_title("R-TT (R value vs Time & Threshold)", fontsize=10)

    # -------- 左下：输入数据 + 阈值 + 异常区填充 + 预报期 + 震例 --------
    ax3 = fig.add_subplot(2, 2, 3)
    dark_purple = (0.25, 0, 0.45)
    ax3.plot(dt, abs_err, "-", color=dark_purple, linewidth=1, label="|Input Data Value|")
    if threshold_mode == 1:
        thr = float(threshold_series[best_threshold_idx]) * std_pd
        ax3.axhline(thr, color="r", linestyle="--", linewidth=1.2, label=f"{threshold_series[best_threshold_idx]:.2f}×STD")
    else:
        thr = float(threshold_series[best_threshold_idx])
        ax3.axhline(thr, color="r", linestyle="--", linewidth=1.2, label=f"Threshold={thr:.2f}")
    data_min = float(np.nanmin(abs_err))
    data_max = float(np.nanmax(abs_err))
    data_range = data_max - data_min
    if data_range <= 0:
        data_range = 1.0
    y_min = data_min
    y_max = data_min + data_range * 1.1
    eq_line_top = data_min + data_range * 1.05
    ax3.set_ylim(y_min, y_max)
    # 异常区填充
    for st, ed in zip(ab_st_best, ab_ed_best):
        mask = (times >= st) & (times <= ed)
        if not np.any(mask):
            continue
        ax3.fill_between(
            np.array(dt)[mask], abs_err[mask], thr,
            color=(250 / 255, 200 / 255, 205 / 255), alpha=0.8,
        )
    # 预报期（段长>预报期则整段，否则从起点起预报期）
    for st, ed in zip(ab_st_best, ab_ed_best):
        if ed - st > best_alarm_day:
            pred_st, pred_ed = float(st), float(ed)
        else:
            pred_st, pred_ed = float(st), float(st) + best_alarm_day
        m = (times >= pred_st) & (times <= pred_ed)
        if not np.any(m):
            continue
        ax3.plot(
            np.array(dt)[m], np.full(np.count_nonzero(m), thr),
            "-", color=(0.0, 0.4, 0.0), linewidth=4,
        )
    # 地震线：从阈值线至 data_range*1.05
    for i in range(len(eq_dt)):
        color = "r" if eq_flags[i] else "b"
        ax3.plot([eq_dt[i], eq_dt[i]], [thr, eq_line_top], "k-", linewidth=0.8)
        ax3.plot(eq_dt[i], eq_line_top, "o", color=color, markerfacecolor=color, markersize=6)
    ax3.set_xlim(dt.min(), dt.max())
    if threshold_mode == 1:
        ax3.set_xlabel(f"{threshold_series[best_threshold_idx]:.2f}×STD   Mag>={mag_min:.1f}")
    else:
        ax3.set_xlabel(f"Threshold={threshold_series[best_threshold_idx]:.2f}   Mag>={mag_min:.1f}")
    ax3.set_ylabel("|Input Data Value|")
    ax3.set_title(f"R={Rmax:.4f}, R0={R0:.4f}, alarm={best_alarm_day} days", fontsize=10, color="r")
    h3, l3 = ax3.get_legend_handles_labels()
    h3 = [hh for hh, ll in zip(h3, l3) if ll not in ("Predicted EQ", "Unpredicted EQ")]
    h3.extend([
        Line2D([0], [0], marker="o", color="w", markerfacecolor="r", markersize=6, label="Predicted EQ"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="b", markersize=6, label="Unpredicted EQ"),
    ])
    ax3.legend(handles=h3, loc="best", fontsize=8)
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for label in ax3.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")
    ax3.grid(True, linestyle=":", alpha=0.5)

    # -------- 右下：Molchan 图 --------
    ax4 = fig.add_subplot(2, 2, 4)
    N_eq = len(eq_times)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.plot([0, 1], [1, 0], "k--", linewidth=1.5, label="Gain=1", zorder=1)
    tau_curve_molchan = None
    nu_curve_molchan = None
    # 2.5% 显著性参考线 (h=1..N)
    if N_eq >= 1:
        tao_ref = []
        nu_ref = []
        for h in range(1, N_eq + 1):
            t = _solve_tao_for_alpha(N_eq, h, 0.025)
            if np.isfinite(t):
                tao_ref.append(t)
                nu_ref.append(1.0 - h / N_eq)
        if len(tao_ref) > 0:
            tao_ref = np.array(tao_ref, dtype=float)
            nu_ref = np.array(nu_ref, dtype=float)
            order_ref = np.argsort(tao_ref)
            tao_ref = tao_ref[order_ref]
            nu_ref = nu_ref[order_ref]
            ax4.plot(tao_ref, nu_ref, color="r", linewidth=1.8, label=r"$\alpha$=2.5%", zorder=2,
                     linestyle=(0, (1.5, 0.8, 0.4, 0.8)))  # 更短的点划线
    if abnorm_x_list is not None and abnorm_y_list is not None and len(abnorm_x_list) > 0:
        dt_start = float(times[0])
        dt_end = float(times[-1])
        tau_curve_molchan, nu_curve_molchan = _molchan_curve_at_alarm_day(
            abnorm_x_list, abnorm_y_list, eq_times, best_alarm_day, dt_start, dt_end
        )
        ax4.plot(tau_curve_molchan, nu_curve_molchan, "b-", linewidth=2, label="Molchan curve", zorder=3)
        ab_sec_best = ab_ed_best - ab_st_best
        _, success_rate_best, occupied_best = r_value_single(
            eq_times, ab_st_best, ab_sec_best, best_alarm_day, dt_start, dt_end
        )
        tau_best = occupied_best
        nu_best = 1.0 - success_rate_best
        ax4.plot(tau_best, nu_best, "o", color="m", markerfacecolor="m", markeredgecolor="k", markersize=8, label="Rmax", zorder=4)
        success_best = int(np.sum(eq_flags))
        gain_best = (success_rate_best / tau_best) if tau_best > 1e-9 else 0.0
        sig_best = _significance_level(N_eq, success_best, tau_best, use_normal=(N_eq >= 20)) if N_eq > 0 else np.nan
        sig_str = f"{float(sig_best) * 100:.2f}%" if np.isfinite(sig_best) else "N/A"
        txt = f"R={Rmax:.4f}  R0={R0:.4f}\nGain={gain_best:.4f}  α={sig_str}"
        ax4.annotate(txt, xy=(tau_best, nu_best), xytext=(0.45, 0.5), fontsize=7,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
                     arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    # AUC：曲线左下方面积，标注在左下角
    if tau_curve_molchan is not None and nu_curve_molchan is not None:
        auc_val = _molchan_auc(tau_curve_molchan, nu_curve_molchan)
        ax4.text(0.05, 0.05, f"AUC={auc_val:.4f}", fontsize=9, transform=ax4.transAxes,
                 verticalalignment="bottom", horizontalalignment="left",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9), zorder=5)
    ax4.set_xlabel(r"Occupancy $\tau$", fontsize=9)
    ax4.set_ylabel(r"Miss rate $\nu$", fontsize=9)
    ax4.set_title("Molchan diagram (alarm=" + str(best_alarm_day) + " d)", fontsize=10)
    ax4.legend(loc="best", fontsize=6)
    ax4.set_ylim(0, 1)
    ax4.set_xlim(0, 1)
    ax4.autoscale(False)
    ax4.set_aspect("equal", adjustable="box")

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ===========================
# 五、主流程
# ===========================


def run_molchan_graph(
    data_file: Path,
    polygon_file: Path,
    eq_catalog_file: Path,
    station_lon: float,
    station_lat: float,
    interference_file: Path | None = None,
    data_has_header: bool = True,
    data_delimiter: str = "\t",
    data_encoding: str = "utf-8",
    mag_min: float = 3.5,
    mag_max: float = 9.0,
    threshold_mode: int = 1,
    rate_min: float = 0.0,
    rate_max: float = 10.0,
    rate_step: float = 0.1,
    thresh_start: float = 50.0,
    thresh_end: float = 180.0,
    thresh_step: float = 1.0,
    alm_day_start: int = 60,
    alm_day_end: int = 720,
    alm_day_step: int = 1,
    use_cartopy_map: bool = False,
) -> None:
    print(f"数据文件: {data_file.resolve()}")
    print(f"规则文件: {polygon_file.resolve()}")
    print(f"地震目录: {eq_catalog_file.resolve()}")

    if not data_file.exists() or not data_file.is_file():
        print("错误：数据文件不存在或不是文件，程序结束。")
        return
    times, residual = read_data_file(
        data_file,
        has_header=data_has_header,
        delimiter=data_delimiter,
        encoding=data_encoding,
    )
    if len(times) == 0 or len(residual) == 0:
        print("错误：数据文件内容为空或解析失败，程序结束。")
        return
    std_pd = float(np.std(residual, ddof=0))
    print("数据读取完成（时间列 + 数值列），直接进行 Molchan-graph 计算。")

    poly_x, poly_y = read_polygon(polygon_file)
    if len(poly_x) == 0:
        print("错误：规则文件（多边形）内容为空，程序结束。")
        return

    cat = EarthquakeCatalog(eq_catalog_file)
    df_all = cat.load()
    if df_all.empty:
        print("错误：地震目录为空，程序结束。")
        return
    t_start = _from_matlab_datenum(float(times[0]))
    t_end = _from_matlab_datenum(float(times[-1]))
    df_sel = cat.select(t_start, t_end, mag_min, mag_max, poly_x, poly_y)
    if df_sel.empty:
        print("错误：在规则与时间、震级范围内未筛选到地震，程序结束。")
        return
    eq_times = _datetime_like_to_julian(df_sel["time"])
    eq_lons = df_sel["lon"].to_numpy()
    eq_lats = df_sel["lat"].to_numpy()
    eq_mags = df_sel["mag"].to_numpy()
    print(f"筛选到 {len(eq_times)} 个震例。")

    interf_start, interf_end = load_interference_periods(interference_file)
    if interference_file and interf_start.size > 0:
        print(f"成功读取 {len(interf_start)} 段干扰时段，将在异常提取和 Molchan-graph 计算中剔除。")
    else:
        if interference_file:
            print("未读取到有效干扰时段，按无干扰处理。")

    if threshold_mode == 1:
        R, abnorm_x_list, abnorm_y_list, alam_days, threshold_series = r_value_grid(
            times, residual, std_pd, eq_times,
            rate_min=rate_min, rate_max=rate_max, rate_step=rate_step,
            alm_day_start=alm_day_start, alm_day_end=alm_day_end, alm_day_step=alm_day_step,
            interf_start=interf_start if interf_start.size > 0 else None,
            interf_end=interf_end if interf_end.size > 0 else None,
        )
    else:
        R, abnorm_x_list, abnorm_y_list, alam_days, threshold_series = r_value_grid_absolute(
            times, residual, eq_times,
            thresh_start=thresh_start, thresh_end=thresh_end, thresh_step=thresh_step,
            alm_day_start=alm_day_start, alm_day_end=alm_day_end, alm_day_step=alm_day_step,
            interf_start=interf_start if interf_start.size > 0 else None,
            interf_end=interf_end if interf_end.size > 0 else None,
        )
    # R 值小于 -1 的替换为 -1
    R = np.maximum(R, -1.0)
    Rmax = float(np.max(R))
    idx = np.argwhere(R == Rmax)[0]
    thresh_idx, day_idx = int(idx[0]), int(idx[1])
    best_threshold = float(threshold_series[thresh_idx])
    best_alm_day = int(alam_days[day_idx])
    ab_st_best = abnorm_x_list[thresh_idx]
    ab_ed_best = abnorm_y_list[thresh_idx]

    # 报准判定：与 is_predicted 一致，预报窗为段长与预报期取长者
    flags = is_predicted(eq_times, ab_st_best, ab_ed_best, best_alm_day)
    success = int(flags.sum())
    miss = len(eq_times) - success
    R0 = get_R0(success, miss)
    # 概率增益、显著性水平、最优预报期下 AUC（与图件一致）
    dt_start = float(times[0])
    dt_end = float(times[-1])
    _, success_rate_best, occupied_best = r_value_single(
        eq_times, ab_st_best, ab_ed_best - ab_st_best, best_alm_day, dt_start, dt_end
    )
    tau_best = occupied_best
    gain_best = (success_rate_best / tau_best) if tau_best > 1e-9 else 0.0
    N_eq = len(eq_times)
    sig_best = _significance_level(N_eq, success, tau_best, use_normal=(N_eq >= 20)) if N_eq > 0 else np.nan
    tau_curve_auc, nu_curve_auc = _molchan_curve_at_alarm_day(
        abnorm_x_list, abnorm_y_list, eq_times, best_alm_day, dt_start, dt_end
    )
    auc_val = _molchan_auc(tau_curve_auc, nu_curve_auc)

    out_dir = data_file.parent
    stem = data_file.stem
    out_txt = out_dir / f"{stem}_result.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("# 基于 Molchan-graph 的预报效能评估结果（单测项）\n")
        f.write(f"# 待评估数据文件: {data_file.name}\n")
        f.write(f"# 地震目录文件: {eq_catalog_file.name}\n")
        f.write(f"# 地震筛选规则文件: {polygon_file.name}\n")
        f.write(f"# 干扰信息文件: {interference_file.name if interference_file else '无'}\n")
        f.write(f"# 台站经纬度: {station_lon:.3f}, {station_lat:.3f}\n")
        f.write("#---------------------------------------------\n")
        jy = "通过检验（R > R0）" if (Rmax > R0 and Rmax > 0) else "未通过检验"
        f.write(f"#------------------------------------ {stem} ({jy}) -----------------------------------\n")
        sig_str_out = f"{float(sig_best) * 100:.2f}" if np.isfinite(sig_best) else "N/A"
        if threshold_mode == 1:
            f.write("# 异常次数   成功预报地震数   漏报地震数    R    R0   概率增益   显著性水平(%)   AUC   最优预报有效期   最优阈值(×STD)  震级下限  震级上限\n")
            f.write(f"{len(ab_st_best):6d} {success:16d} {miss:16d} {Rmax:8.4f} {R0:8.4f} {gain_best:8.4f} {sig_str_out:>8} {auc_val:8.4f} {best_alm_day:4d} {best_threshold:18.2f} {mag_min:8.1f} {mag_max:8.1f}\n")
        else:
            f.write("# 异常次数   成功预报地震数   漏报地震数    R    R0   概率增益   显著性水平(%)   AUC   最优预报有效期   最优阈值   震级下限  震级上限\n")
            f.write(f"{len(ab_st_best):6d} {success:16d} {miss:16d} {Rmax:8.4f} {R0:8.4f} {gain_best:8.4f} {sig_str_out:>8} {auc_val:8.4f} {best_alm_day:4d} {best_threshold:18.2f} {mag_min:8.1f} {mag_max:8.1f}\n")
        f.write("# 具体如下：\n")
        f.write("# 异常开始时间    异常最后时间    是否报准地震   异常距地震发生时间(天)  地震发生时间  地震经度    地震纬度    地震震级    震中距(Km)\n")
        eq_floor = np.floor(eq_times).astype(int)
        for i in range(len(ab_st_best)):
            st, ed = float(ab_st_best[i]), float(ab_ed_best[i])
            # 预报窗结束与 is_predicted 一致：段长>预报期用 ed，否则 st+预报期
            win_ed = st + best_alm_day if (ed - st) <= best_alm_day else ed
            st_str = _from_matlab_datenum(st).strftime("%Y%m%d")
            ed_str = _from_matlab_datenum(ed).strftime("%Y%m%d")  # 异常最后时间 = 段终点
            # 与 is_predicted 一致：按整数日判定是否在预报窗内
            st_int = int(np.ceil(st))
            ed_int = int(np.floor(win_ed))
            eq_in = (eq_floor >= st_int) & (eq_floor <= ed_int)
            if not np.any(eq_in):
                f.write(f"{st_str}  {ed_str}      否\n")
                continue
            for j in np.where(eq_in)[0]:
                t_eq = float(eq_times[j])
                days_st = int(np.floor(t_eq - st))
                days_ed = int(np.floor(t_eq - ed))
                eq_date = _from_matlab_datenum(t_eq).strftime("%Y%m%d")
                dist_km = _distance_km(eq_lons[j], eq_lats[j], station_lon, station_lat)
                f.write(f"{st_str}  {ed_str}      是    {days_st}/{days_ed}  {eq_date}  {eq_lons[j]:7.2f}  {eq_lats[j]:5.2f}  {eq_mags[j]:4.1f}  {dist_km:5.1f}\n")

    out_png = out_dir / f"{stem}_result.png"
    try:
        plot_result_figure(
            times=times,
            residual=residual,
            eq_times=eq_times,
            eq_lons=eq_lons,
            eq_lats=eq_lats,
            eq_flags=flags,
            poly_x=poly_x,
            poly_y=poly_y,
            station_lon=station_lon,
            station_lat=station_lat,
            threshold_series=threshold_series,
            alam_days=alam_days,
            R=R,
            best_threshold_idx=thresh_idx,
            best_alarm_idx=day_idx,
            ab_st_best=ab_st_best,
            ab_ed_best=ab_ed_best,
            best_alarm_day=best_alm_day,
            mag_min=mag_min,
            Rmax=Rmax,
            R0=R0,
            out_png=out_png,
            threshold_mode=threshold_mode,
            std_pd=std_pd,
            use_cartopy_map=use_cartopy_map,
            abnorm_x_list=abnorm_x_list,
            abnorm_y_list=abnorm_y_list,
        )
        print(f"图件已保存: {out_png.resolve()}")
    except Exception as e:
        print(f"绘图异常（不影响数值结果）: {e}")

    print("计算完成，关键结果如下：")
    print(f"  Rmax = {Rmax:.4f}")
    print(f"  R0   = {R0:.4f}")
    if threshold_mode == 1:
        print(f"  最优阈值 = {best_threshold:.2f} × STD")
    else:
        print(f"  最优阈值 = {best_threshold:.2f}")
    print(f"  最优预报期 = {best_alm_day} 天")
    print(f"  成功报准 / 总地震数 = {success} / {len(eq_times)}")
    print(f"结果已写入: {out_txt.resolve()}")


if __name__ == "__main__":
    # ========= 可修改参数 =========
    # test_example 参数：1=cycle异常相关样例；2=trend异常相关样例
    test_example = 2

    if test_example == 1:
        DATA_FILE = Path(r"d:\numerical\Molchan-graph\32016_3_2221_cycle_related_anomaly.txt") # 待评估数据文件
        POLYGON_FILE = Path(r"d:\numerical\Molchan-graph\32016_3_2221_EQ_SelectRule.txt") # 地震筛选规则文件
        EQ_CATALOG_FILE = Path(r"d:\numerical\Molchan-graph\china3msN.eqt") # 地震目录文件
        INTERFERENCE_FILE = Path(r"d:\numerical\Molchan-graph\32016_3_2221_Interference_Period.txt") # 干扰信息文件
        STATION_LON = 120.73 # 台站经度
        STATION_LAT = 31.64 # 台站纬度
        DATA_HAS_HEADER = False # 数据文件是否有表头
        DATA_DELIMITER = "\t" # 数据文件列分隔符
        MAG_MIN = 3.5       # 地震目录筛选下限
        MAG_MAX = 9.0       # 地震目录筛选上限
        ALM_DAY_START = 60  # 预报期起点
        ALM_DAY_END = 720   # 预报期终点
        ALM_DAY_STEP = 1    # 预报期步长
        USE_CARTOPY_MAP = False # 是否使用cartopy地图
        THRESHOLD_MODE = 1  # 1=按数据STD倍数扫描阈值；2=按给定阈值范围（数据单位）生成阈值
        RATE_MIN = 0.0      # 阈值倍数最小值
        RATE_MAX = 10.0     # 阈值倍数最大值
        RATE_STEP = 0.1     # 阈值倍数步长
    elif test_example == 2:
        DATA_FILE = Path(r"d:\numerical\Molchan-graph\14001_1_2231_trend_related_anomaly.txt") # 待评估数据文件
        POLYGON_FILE = Path(r"d:\numerical\Molchan-graph\14001_1_2231_EQ_SelectRule.txt") # 地震筛选规则文件
        EQ_CATALOG_FILE = Path(r"d:\numerical\Molchan-graph\china3ms.eqt") # 地震目录文件
        INTERFERENCE_FILE = Path(r"d:\numerical\Molchan-graph\14001_1_2231_Interference_Period.txt") # 干扰信息文件
        STATION_LON = 112.434 # 台站经度
        STATION_LAT = 37.713 # 台站纬度
        DATA_HAS_HEADER = False # 数据文件是否有表头
        DATA_DELIMITER = "\t" # 数据文件列分隔符
        MAG_MIN = 4.0       # 地震目录筛选下限
        MAG_MAX = 9.0       # 地震目录筛选上限
        ALM_DAY_START = 1   # 预报期起点
        ALM_DAY_END = 360   # 预报期终点
        ALM_DAY_STEP = 1    # 预报期步长
        USE_CARTOPY_MAP = False # 是否使用cartopy地图
        THRESHOLD_MODE = 2  # 1=按数据STD倍数扫描阈值；2=按给定阈值范围（数据单位）生成阈值
        THRESH_START = 50.0   # 模式2：阈值范围起点（针对trend文件可调）
        THRESH_END = 180.0    # 模式2：阈值范围终点
        THRESH_STEP = 1.0    # 模式2：阈值步长        
    else:
        raise ValueError("test_example只能为1或2")

    kwargs = dict(
        data_file=DATA_FILE,
        polygon_file=POLYGON_FILE,
        eq_catalog_file=EQ_CATALOG_FILE,
        station_lon=STATION_LON,
        station_lat=STATION_LAT,
        interference_file=INTERFERENCE_FILE,
        data_has_header=DATA_HAS_HEADER,
        data_delimiter=DATA_DELIMITER,
        mag_min=MAG_MIN,
        mag_max=MAG_MAX,
        threshold_mode=THRESHOLD_MODE,
        alm_day_start=ALM_DAY_START,
        alm_day_end=ALM_DAY_END,
        alm_day_step=ALM_DAY_STEP,
        use_cartopy_map=USE_CARTOPY_MAP,
    )
    if test_example == 1:
        kwargs.update(rate_min=RATE_MIN, rate_max=RATE_MAX, rate_step=RATE_STEP)
    else:
        kwargs.update(thresh_start=THRESH_START, thresh_end=THRESH_END, thresh_step=THRESH_STEP)
    run_molchan_graph(**kwargs)
