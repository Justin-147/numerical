import math
import warnings
from pathlib import Path
from typing import List, Tuple, Dict

# 抑制已知的 UserWarning，避免控制台刷屏
warnings.filterwarnings("ignore", category=UserWarning, message=".*[Ll]evel value of.*too high.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*[Dd]iscarding nonzero nanoseconds.*")

import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.interpolate import PchipInterpolator
from scipy.stats import f as f_dist
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


"""
单测项破年变预报效能分析：读取观测数据与地震目录，经预处理、年变显著性判断、
傅里叶滑动年变拟合与异常提取，在阈值×预测期网格上计算 R 值，输出 Rmax、R0、
报准/漏报统计及图件。使用方法与参数配置见同目录 README_python.md。
"""


# ===========================
# 一、地震目录读取与筛选
# ===========================


class EarthquakeCatalog:
    """
    地震目录读取与筛选：支持 EQT 定长格式，可按时间、震级、多边形范围筛选震例。
    """

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        """
        读取 EQT 格式地震目录，按定长字段解析为 DataFrame。
        """
        if self._df is not None:
            return self._df

        lines = self.file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        records: List[dict] = []
        for line in lines:
            if not line.strip():
                continue
            # EQT 定长字段切片
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

            records.append(
                {
                    "time": pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute, second=second),
                    "lon": lon,
                    "lat": lat,
                    "mag": mag,
                    "depth": depth,
                }
            )

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
        """
        按时间、震级和多边形空间范围筛选震例。
        """
        df = self.load()
        mask_time = (df["time"] >= start_time) & (df["time"] <= end_time)
        mask_mag = (df["mag"] >= mag_min) & (df["mag"] <= mag_max)
        inside = points_in_polygon(df["lon"].to_numpy(), df["lat"].to_numpy(), polygon_lon, polygon_lat)
        mask = mask_time & mask_mag & inside
        return df.loc[mask].reset_index(drop=True)


def points_in_polygon(xs: np.ndarray, ys: np.ndarray, poly_x: np.ndarray, poly_y: np.ndarray) -> np.ndarray:
    """
    射线法判断点是否在多边形内部。
    """
    n_points = xs.shape[0]
    inside = np.zeros(n_points, dtype=bool)
    n_vert = len(poly_x)
    for i in range(n_points):
        x = xs[i]
        y = ys[i]
        j = n_vert - 1
        c = False
        for k in range(n_vert):
            if ((poly_y[k] > y) != (poly_y[j] > y)) and (
                x
                < (poly_x[j] - poly_x[k]) * (y - poly_y[k]) / (poly_y[j] - poly_y[k] + 1e-12)
                + poly_x[k]
            ):
                c = not c
            j = k
        inside[i] = c
    return inside


# ===========================
# 二、观测数据读取与预处理
# ===========================


def read_time_series_txt(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取时间序列观测数据。

    - 两列：第一列为时间标签（yyyymmddhh 或 yyyymmdd），第二列为观测值；
    - 多列：第一列为日期（yyyymmdd），其余 24 列为小时值。
    """
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    m, n = arr.shape
    if n == 2:
        time_raw = arr[:, 0].astype(np.int64)
        data = arr[:, 1].astype(float)
        return time_raw, data

    # n > 2：按日 24 小时数据
    date_col = arr[:, 0].astype(np.int64)
    values = arr[:, 1:]
    time_unhour = np.repeat(date_col, 24)
    data = values.reshape(-1)

    time_labels = time_unhour * 100
    hour = np.tile(np.arange(24, dtype=int), len(date_col))
    time_labels = time_labels + hour
    return time_labels.astype(np.int64), data.astype(float)


def preprocess_missing_and_steps_hourly(obs: np.ndarray) -> np.ndarray:
    """
    缺数插值 + 台阶修正（小时值）。

    - 缺测：999999/99999 用 PCHIP 插值；
    - 开端缺数：若首点为缺测且第一个有效点索引 > 24，则裁掉 1:idx（保留 idx+1 起）；否则裁掉前 24 个；
    - 台阶：相邻差分，mean_square=3*std(dif_value)（样本标准差），再从 i=2 起对超阈值的差分做后段平移。
    """
    obs = obs.astype(float).copy()
    missing_mask = (obs == 999999.0) | (obs == 99999.0)
    valid_idx = np.where(~missing_mask)[0]
    if valid_idx.size == 0:
        return obs

    # PCHIP 插值缺数
    mi_P = np.where(missing_mask)[0]
    m_P = valid_idx
    if len(m_P) >= 2:
        pchip = PchipInterpolator(m_P, obs[m_P])
        obs[mi_P] = pchip(mi_P)
    else:
        obs[mi_P] = np.interp(mi_P, m_P, obs[m_P])

    # 处理开端缺数
    if mi_P.size > 0 and mi_P[0] == 0:  # 首点为缺测
        first_valid_0based = int(valid_idx[0])
        if first_valid_0based > 24:
            # 删去开头无效段，自首个有效点之后保留
            obs = obs[first_valid_0based + 1 :]
        else:
            obs = obs[24:]

    # 台阶检测与修正
    dif = np.diff(obs)
    if dif.size == 0:
        return obs
    mean_square = 3 * np.std(dif, ddof=1)
    for i in range(1, len(obs)):
        d = dif[i - 1]
        if abs(d) > abs(mean_square) and d > 0:
            obs[i:] -= d
        elif abs(d) > abs(mean_square) and d < 0:
            obs[i:] += abs(d)
    return obs


def preprocess_missing_and_steps_daily(obs: np.ndarray) -> np.ndarray:
    """
    日值缺数插值 + 台阶修正。

    - 缺测用 PCHIP 插值；
    - 不裁剪开头（日值已做缺日补齐）；
    - 台阶：3*std(dif_value) 用样本标准差。
    """
    obs = obs.astype(float).copy()
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

    dif = np.diff(obs)
    if dif.size == 0:
        return obs
    mean_square = 3 * np.std(dif, ddof=1)
    for i in range(1, len(obs)):
        d = dif[i - 1]
        if abs(d) > abs(mean_square) and d > 0:
            obs[i:] -= d
        elif abs(d) > abs(mean_square) and d < 0:
            obs[i:] += abs(d)
    return obs


def _interpolate_missing_pchip_only(obs: np.ndarray, missing_val: float = 999999.0) -> np.ndarray:
    """
    仅对缺测（999999/99999）做 PCHIP 插值，不裁剪、不台阶处理。
    用于 Dailymean 中「有缺数时先 LackOfTime 再 DataPreprocess」的插值一步。
    """
    obs = obs.astype(float).copy()
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


def daily_mean_from_hourly(time_labels: np.ndarray, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    整点值求日均值。

    - 起点为 00：若 mod(TimeDate(1),100)~=0，删前 (24-mod(TimeDate(1),100)) 个点；
    - 数据完整性：N1=length/24，N2=首尾日期日历天数；若 N1<N2 则先按 LackOfTime 补全时间轴
      （缺小时填 999999），再对整段做 PCHIP 插值，再按 24 点一块取 nanmean；
    - 否则直接用 tmpt/tmpd，按 j=1:24:N 取 fix(tmpt(j)/100) 与 nanmean(tmpd(j:j+23))。
    """
    time_labels = np.asarray(time_labels, dtype=np.int64)
    data = np.asarray(data, dtype=float)

    # 整点值起点检查：规定起点时刻为 00
    if time_labels[0] % 100 != 0:
        n_drop = 24 - (time_labels[0] % 100)
        n_drop = min(n_drop, len(time_labels))
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
        # 数据非连续：按 LackOfTime 做标准连续时间序列，缺位填 999999，再 PCHIP 插值
        time_full: List[int] = []
        for d in all_days:
            for h in range(24):
                time_full.append(d * 100 + h)
        time_full_arr = np.array(time_full, dtype=np.int64)
        data_full = np.full(len(time_full_arr), 999999.0, dtype=float)
        # 将已有数据填入对应位置
        t_to_val: Dict[int, float] = dict(zip(time_labels.tolist(), data.tolist()))
        for i, t in enumerate(time_full_arr):
            if t in t_to_val:
                data_full[i] = t_to_val[t]
        data_full = _interpolate_missing_pchip_only(data_full)
        # 按 24 小时一块取均值
        n_blocks = len(time_full_arr) // 24
        out_days = (time_full_arr[::24] // 100).astype(np.int64)
        out_means = np.array(
            [float(np.nanmean(data_full[j : j + 24])) for j in range(0, n_blocks * 24, 24)],
            dtype=float,
        )
        return out_days, out_means
    else:
        # 无缺数：tmpt=TimeDate, tmpd=PreObsData，按步长 24 取块求 nanmean
        n_blocks = len(time_labels) // 24
        if n_blocks == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=float)
        use_len = n_blocks * 24
        out_days = np.array(
            [int(time_labels[j] // 100) for j in range(0, use_len, 24)],
            dtype=np.int64,
        )
        # 忽略 NaN 与缺测标记 999999 后求均值
        block_vals = data[:use_len].reshape(n_blocks, 24).astype(float)
        block_vals = np.where((block_vals == 999999.0) | (block_vals == 99999.0), np.nan, block_vals)
        out_means = np.nanmean(block_vals, axis=1).astype(float)
        return out_days, out_means


def complete_days(first_day: int, last_day: int) -> List[int]:
    """
    根据首尾 yyyymmdd 生成连续日期列表。
    """
    start = pd.to_datetime(str(first_day), format="%Y%m%d")
    end = pd.to_datetime(str(last_day), format="%Y%m%d")
    dates = pd.date_range(start, end, freq="D")
    return [int(d.strftime("%Y%m%d")) for d in dates]


def _fill_missing_days_with_flag(days: np.ndarray, data: np.ndarray, flag_value: float = 999999.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    补齐中间缺失的日期，并用给定 flag 标记缺失值。
    """
    days = days.astype(np.int64)
    first_day, last_day = int(days[0]), int(days[-1])
    all_days = complete_days(first_day, last_day)
    day_to_val: Dict[int, float] = {int(d): float(v) for d, v in zip(days, data)}

    filled_vals: List[float] = []
    for d in all_days:
        v = day_to_val.get(d, flag_value)
        filled_vals.append(v)

    return np.asarray(all_days, dtype=np.int64), np.asarray(filled_vals, dtype=float)


# ===========================
# 三、小波滤波 + 年变显著性
# ===========================


def wavelet_separate_daily(daily_series: np.ndarray, wave_step: int = 9, wavelet: str = "db5") -> Tuple[np.ndarray, np.ndarray]:
    """
    使用离散小波分解求趋势项与高频项。
    （保留作为通用接口，主流程的年变分析使用 `wavelet_filter_for_annual`。）
    """
    import pywt

    coeffs = pywt.wavedec(daily_series, wavelet, level=wave_step)
    approx = pywt.upcoef("a", coeffs[0], wavelet, level=wave_step, take=len(daily_series))
    high = daily_series - approx
    return approx, high


def wavelet_filter_for_annual(daily_series: np.ndarray, level: int = 9, wavelet: str = "db5") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    小波分解后去除近似分量得到年变分析用序列，并去除 1–4 阶细节得到平滑序列：

    - [c,l] = wavedec(PObsData, 9, 'db5')；L_data = wrcoef('a', c, l, 'db5', 9)；H_data_or = PObsData - L_data；
    - H_data = H_data_or；for iw=1:4, D = wrcoef('d', c, l, 'db5', 5-iw)；H_data = H_data - D；end

    近似与细节重构使用 waverec，边界采用对称延拓（mode='sym'）。
    """
    import pywt

    series = np.asarray(daily_series, dtype=float)
    n = len(series)
    # 对称延拓
    coeffs = pywt.wavedec(series, wavelet, level=level, mode="sym")
    # L_data = wrcoef('a',...)：仅保留 cA9、其余层置零再做多级逆变换，用 waverec 实现
    coeffs_approx = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    approx = np.asarray(pywt.waverec(coeffs_approx, wavelet), dtype=float)
    if len(approx) > n:
        approx = approx[:n]
    elif len(approx) < n:
        approx = np.pad(approx, (0, n - len(approx)), mode="constant", constant_values=0.0)
    high_orig = series - approx

    high_smoothed = high_orig.copy()
    # 依次减去 D4、D3、D2、D1 细节
    for iw in range(1, 5):
        k = 5 - iw  # k = 4,3,2,1 → D4, D3, D2, D1
        idx = level + 1 - k  # coeffs[0]=cA9, coeffs[1]=cD9,..., coeffs[9]=cD1
        if idx < 1 or idx >= len(coeffs):
            continue
        detail_coeffs = [np.zeros_like(c) for c in coeffs]
        detail_coeffs[idx] = coeffs[idx].copy()
        Dk = np.asarray(pywt.waverec(detail_coeffs, wavelet), dtype=float)
        if len(Dk) > n:
            Dk = Dk[:n]
        elif len(Dk) < n:
            Dk = np.pad(Dk, (0, n - len(Dk)), mode="constant", constant_values=0.0)
        high_smoothed -= Dk

    return approx, high_orig, high_smoothed


def fourier_slide_annual_3y(residual: np.ndarray, period: int = 365) -> Tuple[np.ndarray, float]:
    """
    三年滑动窗口傅里叶年变拟合，得到拟合曲线与残差：

    - 滑动窗：i=wid..n，ai = sum(emdat(j)*cos(2*pi*(j-i+wid)/T))，fs_y(i)=2*ai/wid，余弦自变量为 1..wid；
    - 前 wid-1 点：用前 wid 点做 mj=wid/T 阶基波拟合，L=wid-1，fs_y(1:wid-1)=y_fit(1:wid-1)；
    - std(err_data,1) 为总体标准差（ddof=0）。
    """
    T = period
    wid = 3 * T
    n = len(residual)
    if n < wid:
        err = residual.copy()
        return err, float(np.std(err, ddof=0))

    fs_y = np.zeros_like(residual, dtype=float)

    # 富氏滑动：for i = wid : length(emdat)（1-based），即设置第 wid..n 个点；0-based 为 i=wid-1..n-1
    # ai = sum(emdat(j)*cos(2*pi*(j-i+wid)/T))，窗内 k=1..wid 对应 cos(2*pi*k/T)
    for i in range(wid - 1, n):
        window = residual[i - wid + 1 : i + 1]
        k = np.arange(1, wid + 1, dtype=float)
        ai = float(np.dot(window, np.cos(2 * math.pi * k / T)))
        fs_y[i] = 2 * ai / wid

    # 基波拟合：mj=wid/T，nh_y=emdat(1:wid)，xl=1:wid，L=wid-1，fs_y(1:wid-1)=y_fit(1:wid-1)
    nh_y = residual[:wid]
    xl = np.arange(1, wid + 1, dtype=float)
    L = xl[-1] - xl[0]  # wid - 1
    mj = int(wid / T)

    bn = np.zeros(mj)
    an = np.zeros(mj)
    for nn in range(1, mj + 1):
        bn[nn - 1] = (2.0 / L) * float(np.trapz(nh_y * np.sin(2 * nn * math.pi * xl / L), xl))
        an[nn - 1] = (2.0 / L) * float(np.trapz(nh_y * np.cos(2 * nn * math.pi * xl / L), xl))
    a0 = (1.0 / L) * float(np.trapz(nh_y, xl))

    x_fit = np.linspace(xl[0], xl[-1], len(nh_y))
    y_fit = np.full_like(x_fit, a0)
    for nn in range(1, mj + 1):
        y_fit += bn[nn - 1] * np.sin(2 * nn * math.pi * x_fit / L) + an[nn - 1] * np.cos(2 * nn * math.pi * x_fit / L)

    # 前 wid-1 个点用基波拟合；第 wid 个点已在上面滑动中赋值（i=wid-1）
    fs_y[: wid - 1] = y_fit[: wid - 1]

    err = residual - fs_y
    std_pd = float(np.std(err, ddof=0))
    return err, std_pd


def detect_annual_significance(data: np.ndarray) -> Tuple[bool, int]:
    """
    年变显著性判别（FFT 频谱 + F 检验与主峰判据）：

    - FFT 得频谱，mag1 = mag_fft*2/Nfft，年变频段 1/425 ~ 1/300；
    - mag_ann 为年变带内幅值，mag_uann = setxor(mag1, mag_ann) 为其余（对称差集）；
    - h1：对 log10(mag1) 与 log10(mag_uann) 做 F 检验（右尾，alpha=0.15），与 vartest2(log10(mag1),log10(mag_uann),...) 一致；
    - h2：主峰（log10(mag1) 最大）是否落在年变带内，是则 1 否则 0；
    - 返回 (是否显著, isAnn)，isAnn 为 0/1/2。
    """
    n = len(data)
    if n < 10:
        return False, 0

    # 单边频谱与幅值 mag1
    fs = 1.0
    y = fft(data, n)
    mag = np.abs(y)
    f = np.arange(n, dtype=float) * fs / n
    half = int(np.fix(n / 2))  # fix(Nfft/2)
    f_fft = f[:half]
    mag_fft = mag[:half]
    mag1 = (mag_fft * 2.0) / n

    # 年变带内幅值 mag_ann，带外用对称差集得 mag_uann
    idx_ann = np.where((f_fft >= 1.0 / 425.0) & (f_fft <= 1.0 / 300.0))[0]
    if idx_ann.size <= 1:
        return False, 0

    mag_ann = mag1[idx_ann]
    mag_uann = np.setxor1d(mag1, mag_ann)

    # F 检验：log10(mag1) 与 log10(mag_uann)，右尾
    log_mag1 = np.log10(mag1)
    log_uann = np.log10(mag_uann)

    h1 = 0
    n1 = log_mag1.size
    n2 = log_uann.size
    if n1 > 1 and n2 > 1:
        var1 = float(np.var(log_mag1, ddof=1))
        var2 = float(np.var(log_uann, ddof=1))
        if var2 > 0:
            F = var1 / var2
            alpha = 0.15
            try:
                Fcrit = float(f_dist.ppf(1.0 - alpha, n1 - 1, n2 - 1))
                h1 = 1 if F > Fcrit else 0
            except Exception:
                h1 = 1 if F > 1.5 else 0

    # h2：主峰是否落在年变带内
    idxmx = np.where(log_mag1 == np.max(log_mag1))[0]
    in_band = (f_fft[idxmx] >= 1.0 / 425.0) & (f_fft[idxmx] <= 1.0 / 300.0)
    h2 = 1 if np.all(in_band) else 0

    is_ann = h1 + h2
    return (is_ann > 0, is_ann)


# ===========================
# 四、破年变异常与 R 值计算
# ===========================


def detect_anomalies_threshold(
    times: np.ndarray,
    residual: np.ndarray,
    std_pd: float,
    rate: float,
    interf_start: np.ndarray | None = None,
    interf_end: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    按阈值 rate×std_pd 提取破年变异常段（残差绝对值超阈值连续区间）。

    扫描规则：
    - 超过阈值：若无开始则记录开始位置/时间，并累计异常点数；
    - 首次低于阈值：记录段终点为最后仍超阈值的下标（i-1），使 time_ed 为异常最后时间；
    - 若异常持续到序列末尾：将最后一个时间/索引作为该段结束。
    幅度取段内残差（未取绝对值）的最大值：max(residual[idx_st:idx_ed+1])。

    推荐用法：不传 interf_*，先得到原始异常段，再用 _del_itf_time 做干扰剔除/裁剪。
    返回：异常开始时间、异常最后时间（段终点）、持续时间、异常幅度。
    """
    abs_err = np.abs(residual)
    n = len(residual)
    idx_st: List[int] = []
    idx_ed: List[int] = []
    pnt_st = False
    num_all = 0

    for i in range(n):
        if abs_err[i] > rate * std_pd:
            num_all += 1
            if not pnt_st:
                idx_st.append(i)
                pnt_st = True
        elif pnt_st:
            # 首次低于阈值：段终点 = 最后仍超阈值的下标（i 为首次低于阈值的下标）
            idx_ed.append(i - 1)
            pnt_st = False
        if i == n - 1 and pnt_st:
            # 异常持续到序列末尾，结束索引为最后一格
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

    # 若提供干扰时段信息，则剔除落在干扰区间内的异常
    if interf_start is not None and interf_end is not None and len(interf_start) > 0:
        keep_mask = np.ones(len(time_st), dtype=bool)
        for j in range(len(interf_start)):
            st_j = float(interf_start[j])
            ed_j = float(interf_end[j])
            # 只要异常区间与干扰区间有重叠即剔除
            overlap = ~((time_ed < st_j) | (time_st > ed_j))
            keep_mask &= ~overlap

        time_st = time_st[keep_mask]
        time_ed = time_ed[keep_mask]
        time_section = time_section[keep_mask]
        idx_st_arr = idx_st_arr[keep_mask]
        idx_ed_arr = idx_ed_arr[keep_mask]

    # 幅度为段内残差（未取绝对值）的最大值
    abn_ampl = np.array(
        [float(np.max(residual[idx_st_arr[i] : idx_ed_arr[i] + 1])) for i in range(len(idx_st_arr))],
        dtype=float,
    )
    _ = num_all - np.sum(time_section)
    return time_st, time_ed, time_section, abn_ampl


def _del_itf_time(
    time_st: np.ndarray,
    time_ed: np.ndarray,
    itf_st: np.ndarray,
    itf_ed: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    剔除或裁剪与干扰时段相交的异常段：判交时始终用原始异常段边界；
    若干扰开始早于异常开始则整段剔除，否则将段终点（异常最后时间）裁为干扰开始时间。
    返回剔除或裁剪后的 (timeST, timeED)。
    """
    if len(time_st) == 0:
        return time_st, time_ed
    st = time_st.astype(float).copy()
    ed = time_ed.astype(float).copy()
    for i in range(len(st)):
        orig_st = float(st[i])
        orig_ed = float(ed[i])
        for j in range(len(itf_st)):
            # 相交条件等价于 intersect(AbsT{i}, itfT{j}) 非空；用原始段判交
            if orig_st <= itf_ed[j] and orig_ed >= itf_st[j]:
                if itf_st[j] < orig_st:
                    st[i] = np.nan
                    ed[i] = np.nan
                    break
                else:
                    ed[i] = float(itf_st[j])
    keep = ~(np.isnan(st) | np.isnan(ed))
    return st[keep], ed[keep]


def r_value_single(
    eq_times: np.ndarray,
    alarm_start: np.ndarray,
    alarm_sec: np.ndarray,
    alarm_days: int,
    dt_start: float,
    dt_end: float,
) -> Tuple[float, float, float]:
    """
    针对单一预测期计算 R 值、报准率和时间占用率；报准判定复用 is_predicted。
    """
    n_eq = max(len(eq_times), 1)
    dt_span = max(float(dt_end - dt_start), 1e-6)
    if len(alarm_start) == 0:
        return 0.0, 0.0, 0.0

    # 每段预警结束 = 开始+持续 或 开始+预报期（取长者）
    ab_ed = np.array(
        [
            st + (sec if sec > alarm_days else alarm_days)
            for st, sec in zip(alarm_start, alarm_sec)
        ],
        dtype=float,
    )
    flags = is_predicted(eq_times, alarm_start, ab_ed, alarm_days)
    success = int(np.sum(flags))
    success_rate = success / n_eq
    total_len = float(np.sum(ab_ed - alarm_start))
    occupied_rate = total_len / dt_span
    r_val = success_rate - occupied_rate
    return r_val, success_rate, occupied_rate


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
    """
    在 (阈值倍数, 预测期) 网格上计算 R 矩阵。返回 R, abnorm_X, abnorm_Y, alamDaysArr, rate_Series。
    有干扰时先按阈值提取异常再按干扰时段剔除/裁剪；无干扰时直接使用异常起止。
    rate 上限由 max(|残差|)/std_pd 自动截断；rate_series 不包含末端浮点多出的点。
    """
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
        # 先按阈值提取异常，再统一做干扰剔除/裁剪
        time_st, time_ed, time_sec, _amp = detect_anomalies_threshold(
            times, residual, std_pd, rate, interf_start=None, interf_end=None
        )
        if has_interf:
            abn_st, abn_ed = _del_itf_time(time_st, time_ed, interf_start, interf_end)
            abn_sec = abn_ed - abn_st
        else:
            abn_st = time_st
            abn_ed = time_ed
            abn_sec = time_sec
        # rate 足够大导致无异常时，后续 rate 更大必然也无异常，提前结束并截断
        if len(abn_st) == 0:
            break
        abnorm_x_list.append(abn_st)
        abnorm_y_list.append(abn_ed)
        for j, alm_day in enumerate(alm_days_arr):
            r_val, _, _ = r_value_single(
                eq_times, abn_st, abn_sec, alm_day, dt_start, dt_end
            )
            R_all[i, j] = r_val

    n_valid = len(abnorm_x_list)
    R_all = R_all[:n_valid, :]
    rate_series = rate_series[:n_valid]
    return R_all, abnorm_x_list, abnorm_y_list, alm_days_arr, rate_series


def get_R0(success_count: int, miss_count: int) -> float:
    """
    由报准数与漏报数计算 R0 阈值（累积二项分布近似）。
    """
    k = success_count
    n = success_count + miss_count
    if n == 0:
        return 0.0
    alpha = 0.025
    min_alpha = 1e38
    min_alpha_p = 0.0
    from math import comb

    for P in np.linspace(0.0, 1.0, 1001):
        s = 0.0
        for i in range(k, n + 1):
            s += comb(n, i) * (P**i) * ((1 - P) ** (n - i))
        diff = abs(s - alpha)
        if diff < min_alpha:
            min_alpha = diff
            min_alpha_p = P
    return k / n - min_alpha_p


def is_predicted(eq_times: np.ndarray, ab_st: np.ndarray, ab_ed: np.ndarray, alarm_day: int) -> np.ndarray:
    """
    判断每个地震是否落在任一预警时段内（报准）。
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


# ===========================
# 五、绘图
# ===========================


# 内部时间历元：1970-01-01 对应 719529（日数）
_MATLAB_DATENUM_1970 = 719529


def _to_matlab_datenum(ts: pd.Timestamp) -> float:
    """将 pandas Timestamp 转为内部日数（自 1970-01-01）。"""
    delta = (ts - pd.Timestamp("1970-01-01")).total_seconds() / 86400.0
    return _MATLAB_DATENUM_1970 + delta


def _from_matlab_datenum(d: float) -> pd.Timestamp:
    """将内部日数转为 pandas Timestamp。"""
    return pd.Timestamp("1970-01-01") + pd.Timedelta(days=float(d) - _MATLAB_DATENUM_1970)


def _distance_km(lon_a: float, lat_a: float, lon_b: float, lat_b: float) -> float:
    """球面两点距离（km），输入为度。"""
    rad = np.pi / 180.0
    la, lb = lat_a * rad, lat_b * rad
    dlon = (lon_a - lon_b) * rad
    c = np.sin(la) * np.sin(lb) + np.cos(la) * np.cos(lb) * np.cos(dlon)
    c = np.clip(c, -1.0, 1.0)
    return float(6371.137 * np.arccos(c))


def _to_datetime_from_julian(times: np.ndarray) -> np.ndarray:
    """
    将内部时间（日数）转为 datetime 数组，用于画图。
    """
    return np.array([_from_matlab_datenum(t).to_pydatetime() for t in np.atleast_1d(times).ravel()])


def _datetime_like_to_julian(arr) -> np.ndarray:
    """
    将日期/时间转为内部日数（自 1970-01-01）。
    输入可为 datetime、时间码数组或 Series/Index。
    """
    dt = pd.to_datetime(arr)
    if pd.api.types.is_scalar(dt):
        return np.array([_to_matlab_datenum(pd.Timestamp(dt))], dtype=float)
    return np.asarray([_to_matlab_datenum(pd.Timestamp(t)) for t in dt], dtype=float)


def _parse_time_code_to_timestamp(code: int) -> pd.Timestamp:
    """
    将整数时间码（yyyymmdd、yyyymmddhh、yyyymmddhhMMSS 等）转为 pandas Timestamp。
    """
    s = str(int(code))
    if len(s) == 14:
        year = int(s[0:4])
        month = int(s[4:6])
        day = int(s[6:8])
        hour = int(s[8:10])
        minute = int(s[10:12])
        second = int(s[12:14])
    elif len(s) == 12:
        year = int(s[0:4])
        month = int(s[4:6])
        day = int(s[6:8])
        hour = int(s[8:10])
        minute = int(s[10:12])
        second = 0
    elif len(s) == 10:
        year = int(s[0:4])
        month = int(s[4:6])
        day = int(s[6:8])
        hour = int(s[8:10])
        minute = 0
        second = 0
    elif len(s) == 8:
        year = int(s[0:4])
        month = int(s[4:6])
        day = int(s[6:8])
        hour = minute = second = 0
    elif len(s) == 6:
        year = int(s[0:4])
        month = int(s[4:6])
        day = 1
        hour = minute = second = 0
    elif len(s) == 4:
        year = int(s[0:4])
        month = day = 1
        hour = minute = second = 0
    else:
        # 无法识别的长度，退化为按日期解析（可能抛异常）
        return pd.to_datetime(s)

    return pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute, second=second)


def load_interference_periods(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取干扰信息文件，返回干扰起止时间（儒略日）。

    仅支持文本文件：首行为标题，其后两列分别为「干扰影响开始时间」「结束时间」，
    时间码格式同 _parse_time_code_to_timestamp（如 yyyymmdd、yyyymmddhh 等）。
    文件不存在或为空时返回空数组。
    """
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


def plot_result_figure(
    times: np.ndarray,
    series: np.ndarray,
    low_freq: np.ndarray,
    err: np.ndarray,
    eq_times: np.ndarray,
    eq_lons: np.ndarray,
    eq_lats: np.ndarray,
    eq_flags: np.ndarray,
    poly_x: np.ndarray,
    poly_y: np.ndarray,
    station_lon: float,
    station_lat: float,
    rate_series: np.ndarray,
    alam_days: np.ndarray,
    R: np.ndarray,
    best_rate_idx: int,
    best_alarm_idx: int,
    ab_st_best: np.ndarray,
    ab_ed_best: np.ndarray,
    best_alarm_day: int,
    mag_start: float,
    Rmax: float,
    R0: float,
    out_png: Path,
    use_cartopy_map: bool = False,
) -> None:
    """
    四幅子图：① 滤波年变序列 + 拟合年变曲线；② 空间分布（台站、多边形、报准/漏报震例，可选 cartopy 底图）；
    ③ 残差 + 阈值 + 异常区填充 + 预警段 + 震例（红=报准、蓝=漏报）；④ R-TT 曲面（预测期×阈值）。
    """
    # 转为 datetime 方便画图
    dt = _to_datetime_from_julian(times)
    eq_dt = _to_datetime_from_julian(eq_times)

    fig = plt.figure(figsize=(11, 7), dpi=120)
    fig.suptitle("Cycle-related anomaly", fontsize=14)

    # -------- Subplot 1: 滤波年变序列 + 拟合年变曲线 --------
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(dt, series, "k-", linewidth=1, label="Filtered Annual")
    ax1.plot(dt, low_freq, "r--", linewidth=1, label="Fitted Annual")
    ax1.set_xlabel("Time / year")
    ax1.set_ylabel("Observation")
    ax1.set_xlim(dt.min(), dt.max())
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for label in ax1.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")

    # -------- Subplot 2: 空间分布（多边形±0.5 范围，可选 cartopy 底图） --------
    lon_min, lon_max = float(np.min(poly_x)) - 0.5, float(np.max(poly_x)) + 0.5
    lat_min, lat_max = float(np.min(poly_y)) - 0.5, float(np.max(poly_y)) + 0.5
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

    # 报准与漏报震例均绘制，图例中始终显示 Predicted EQ / Unpredicted EQ（含漏报为 0 的情况）
    pred_idx = np.where(eq_flags)[0]
    miss_idx = np.where(~eq_flags)[0]
    if pred_idx.size > 0:
        _plot_ax2(eq_lons[pred_idx], eq_lats[pred_idx], "ro", markersize=6)
    if miss_idx.size > 0:
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

    # -------- Subplot 4: R-TT 曲面（色标 [-1,1]，最优点标黑） --------
    ax4 = fig.add_subplot(2, 2, 4)
    X, Y = np.meshgrid(alam_days, rate_series)
    c = ax4.pcolormesh(X, Y, R, shading="auto", cmap="jet", vmin=-1.0, vmax=1.0)
    fig.colorbar(c, ax=ax4, label="R value")
    ax4.plot(
        alam_days[best_alarm_idx],
        rate_series[best_rate_idx],
        "ko",
        markersize=8,
        markerfacecolor="k",
    )
    ax4.set_xlabel("Prediction time / days")
    ax4.set_ylabel("Threshold (×STD)")
    ax4.set_title("R-TT (R value vs Time & Threshold)", fontsize=10)

    # -------- Subplot 3: 残差 + 阈值 + 异常区填充 + 预报期 + 震例（风格与 R 程序一致）--------
    ax3 = fig.add_subplot(2, 2, 3)
    abs_err = np.abs(err)
    dark_purple = (0.25, 0, 0.45)
    ax3.plot(dt, abs_err, "-", color=dark_purple, linewidth=1, label="|Residual|")

    # 阈值线：最优 rate × 残差总体标准差，虚线
    thr = float(rate_series[best_rate_idx]) * float(np.std(err, ddof=0))
    ax3.axhline(thr, color="r", linestyle="--", linewidth=1.2, label=f"{rate_series[best_rate_idx]:.2f}×STD")

    # Y 轴范围：与 R 程序一致，data_min 到 data_min + data_range*1.1
    data_min = float(np.nanmin(abs_err))
    data_max = float(np.nanmax(abs_err))
    data_range = data_max - data_min
    if data_range <= 0:
        data_range = 1.0
    y_min = data_min
    y_max = data_min + data_range * 1.1
    eq_line_top = data_min + data_range * 1.05
    ax3.set_ylim(y_min, y_max)

    # 异常区粉色填充
    for st, ed in zip(ab_st_best, ab_ed_best):
        mask = (times >= st) & (times <= ed)
        if not np.any(mask):
            continue
        ax3.fill_between(
            np.array(dt)[mask],
            abs_err[mask],
            thr,
            color=(250 / 255, 200 / 255, 205 / 255),
            alpha=0.8,
        )

    # 预报期：段长>预报期则整段，否则从起点起预报期，深绿色粗线（与 R 程序一致）
    dark_green = (0.0, 0.4, 0.0)
    for st, ed in zip(ab_st_best, ab_ed_best):
        if ed - st > best_alarm_day:
            pred_st, pred_ed = float(st), float(ed)
        else:
            pred_st, pred_ed = float(st), float(st) + best_alarm_day
        m = (times >= pred_st) & (times <= pred_ed)
        if not np.any(m):
            continue
        ax3.plot(
            np.array(dt)[m],
            np.full(np.count_nonzero(m), thr),
            "-", color=dark_green, linewidth=4,
        )

    # 地震标注：竖线从阈值到 eq_line_top，顶端圆点（与 R 程序一致）
    for i in range(len(eq_dt)):
        color = "r" if eq_flags[i] else "b"
        ax3.plot([eq_dt[i], eq_dt[i]], [thr, eq_line_top], "k-", linewidth=0.8)
        ax3.plot(eq_dt[i], eq_line_top, "o", color=color, markerfacecolor=color, markersize=6)

    ax3.set_xlim(dt.min(), dt.max())
    ax3.set_xlabel(f"{rate_series[best_rate_idx]:.2f}×STD   Mag>={mag_start:.1f}")
    ax3.set_ylabel("|Residual|")
    ax3.set_title(f"R={Rmax:.4f}, R0={R0:.4f}, alarm={best_alarm_day} days", fontsize=10, color="r")
    from matplotlib.lines import Line2D
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
    interference_file: Path | None = None,
    mag_min: float = 3.5,
    mag_max: float = 9.0,
    rate_min: float = 0.0,
    rate_max: float = 10.0,
    rate_step: float = 0.1,
    alm_day_start: int = 60,
    alm_day_end: int = 720,
    alm_day_step: int = 1,
    use_cartopy_map: bool = False,
) -> None:
    """
    单测项主流程：读取观测数据并预处理，小波分解与年变显著性判断，傅里叶滑动年变拟合；
    在 (阈值, 预测期) 网格上计算 R 矩阵并取 Rmax，结合地震目录计算 R0，输出结果文件与图件。
    """
    print(f"数据文件: {data_file}")
    print(f"规则文件: {polygon_file}")
    print(f"地震目录: {eq_catalog_file}")

    # 1. 观测数据读取
    time_raw, obs_raw = read_time_series_txt(data_file)
    if len(time_raw) == 0 or len(obs_raw) == 0:
        print("错误：观测数据文件（DATA_FILE）内容为空，无法继续，程序结束。")
        return

    # 判断是整点值（10 位 yyyymmddhh）还是日值（8 位 yyyymmdd）
    time_len = len(str(int(time_raw[0])))
    if time_len == 10:
        # 整点值：缺数/台阶预处理 + 转日均值
        obs_pre = preprocess_missing_and_steps_hourly(obs_raw)
        days, day_vals = daily_mean_from_hourly(time_raw, obs_pre)
        times = _datetime_like_to_julian(pd.to_datetime(days.astype(str), format="%Y%m%d"))
        series = day_vals
        print("数据预处理完成（整点值→日均值，缺数与台阶已处理）")
    elif time_len == 8:
        # 日值：先按首尾日期补齐缺失日（缺日填 999999），再缺数/台阶预处理（对应 LackOfTime + DataPreprocess2）
        days_tmp = time_raw.astype(np.int64)
        days_filled, obs_flagged = _fill_missing_days_with_flag(days_tmp, obs_raw)
        obs_pre = preprocess_missing_and_steps_daily(obs_flagged)
        days = days_filled
        times = _datetime_like_to_julian(pd.to_datetime(days.astype(str), format="%Y%m%d"))
        series = obs_pre
        print("数据预处理完成（日值，缺日补齐 + 缺数与台阶已处理）")
    else:
        print("错误：当前程序仅支持整点值（时间列为 10 位 yyyymmddhh）或日值（8 位 yyyymmdd）。")
        print("请提供整点值数据（yyyymmddhh）或日值数据（yyyymmdd），然后重新运行。")
        return

    # 2. 小波滤波，得到年变分析用序列
    approx, high_orig, high = wavelet_filter_for_annual(series)
    print("小波滤波完成，得到 H_data_or 与去除 1–4 阶高频后的 H_data")

    # 3. 年变显著性判断并输出 isAnn
    is_significant, is_ann = detect_annual_significance(high)
    print(f"年变显著性 isAnn = {is_ann}（0=无显著年变，1 或 2=有年变）")
    if not is_significant:
        print("序列无显著年变，不继续计算 R 值。")
        # 仍输出结果文件说明“无显著年变”
        out_path = data_file.with_suffix(".python_result.txt")
        with out_path.open("w", encoding="utf-8") as f:
            f.write("# Python 破年变预报效能结果（单测项）\n")
            f.write(f"# 数据文件: {data_file.name}\n")
            f.write(f"# 规则文件: {polygon_file.name}\n")
            f.write(f"# 台站经纬度: {station_lon:.3f}, {station_lat:.3f}\n")
            f.write(f"# 地震目录文件: {eq_catalog_file.name}\n")
            f.write("#---------------------------------------------\n")
            f.write(f"isAnn = {is_ann}\n")
            f.write("该测项序列无显著年变，未进行 R 值计算。\n")
        print(f"无显著年变结果已写入文件: {out_path}")
        return
    print("年变显著，开始傅里叶滑动年变拟合")

    # 4. 傅里叶滑动年变拟合，得到残差与标准差（输入为 H_data）
    err, std_pd = fourier_slide_annual_3y(high)
    print(f"傅里叶滑动拟合完成，残差 STD = {std_pd:.4f}")

    fitted_annual = high - err

    # 5. 读取震例对应规则（多边形）：首行为标题（可能含中文），其后两列为经度、纬度
    try:
        with open(polygon_file, encoding="utf-8") as f:
            poly = np.loadtxt(f, skiprows=1)
    except UnicodeDecodeError:
        with open(polygon_file, encoding="gbk") as f:
            poly = np.loadtxt(f, skiprows=1)
    if poly.ndim == 1:
        poly = poly.reshape(1, -1)
    poly_x = poly[:, 0]
    poly_y = poly[:, 1]
    if len(poly_x) == 0:
        print("错误：规则文件（POLYGON_FILE）内容为空（无有效顶点），无法继续，程序结束。")
        return

    # 6. 读取地震目录，并按时间/震级/空间规则筛选
    cat = EarthquakeCatalog(eq_catalog_file)
    df_eq_all = cat.load()
    if df_eq_all.empty:
        print("错误：地震目录文件（EQ_CATALOG_FILE）内容为空，无法继续，程序结束。")
        return
    t_start = pd.to_datetime(str(int(days[0])), format="%Y%m%d")
    t_end = pd.to_datetime(str(int(days[-1])), format="%Y%m%d")

    df_sel = cat.select(
        start_time=t_start,
        end_time=t_end,
        mag_min=mag_min,
        mag_max=mag_max,
        polygon_lon=poly_x,
        polygon_lat=poly_y,
    )

    if df_sel.empty:
        print("错误：在规则范围与时间、震级条件下未筛选到任何地震记录，无法计算 R 值，程序结束。")
        return

    # 兼容不同 pandas 版本的儒略日转换
    time_series = df_sel["time"]
    eq_times = _datetime_like_to_julian(time_series)
    eq_lons = df_sel["lon"].to_numpy()
    eq_lats = df_sel["lat"].to_numpy()
    eq_mags = df_sel["mag"].to_numpy()
    print(f"筛选到 {len(eq_times)} 个震例。")

    # 7. 干扰信息：无干扰文件或文件内容为空时按无干扰处理
    interf_start_jd: np.ndarray | None = None
    interf_end_jd: np.ndarray | None = None
    if interference_file is not None:
        interf_start_jd, interf_end_jd = load_interference_periods(interference_file)
        if interf_start_jd.size == 0:
            print("未读取到有效干扰时段（文件不存在、为空或格式不符），按无干扰处理。")
            interf_start_jd = None
            interf_end_jd = None
        else:
            print(f"成功读取 {len(interf_start_jd)} 段干扰时段，将在异常提取和 R 值计算中剔除这些时段。")

    # 8. 在 (阈值,预测期) 网格上计算 R 矩阵
    R, abnorm_x_list, abnorm_y_list, alam_days, rate_series = r_value_grid(
        times,
        err,
        std_pd,
        eq_times,
        rate_min=rate_min,
        rate_max=rate_max,
        rate_step=rate_step,
        alm_day_start=alm_day_start,
        alm_day_end=alm_day_end,
        alm_day_step=alm_day_step,
        interf_start=interf_start_jd,
        interf_end=interf_end_jd,
    )

    # 9. 寻找 Rmax 及对应阈值、预测期
    # 将 R < -1 的值裁剪为 -1，避免极端情况下时间占用率过大导致 R 值过小
    R = np.maximum(R, -1.0)
    Rmax = float(np.max(R))
    idx = np.argwhere(R == Rmax)[0]
    rate_idx, day_idx = int(idx[0]), int(idx[1])
    best_rate = float(rate_series[rate_idx])
    best_alm_day = int(alam_days[day_idx])
    ab_st_best = abnorm_x_list[rate_idx]
    ab_ed_best_from_grid = abnorm_y_list[rate_idx]

    # 10. 用最优阈值对应的异常起止计算 R0
    flags = is_predicted(eq_times, ab_st_best, ab_ed_best_from_grid, best_alm_day)
    success = int(flags.sum())
    miss = int(len(eq_times) - success)
    R0 = get_R0(success, miss)

    # 10b. 输出处理序列结果文件（五列：Time, Filtered_Annual, Fitted_Annual, Residual, Anomaly）
    # Anomaly=1：满足最优阈值判定（|残差|>最优阈值）且不在干扰时段内；否则 0
    processed_path = data_file.with_name(data_file.stem + "_processed.txt")
    thr_best = best_rate * std_pd
    in_interf = np.zeros(len(times), dtype=bool)
    if interf_start_jd is not None and interf_end_jd is not None and len(interf_start_jd) > 0:
        for k in range(len(interf_start_jd)):
            in_interf |= (times >= interf_start_jd[k]) & (times <= interf_end_jd[k])
    with open(processed_path, "w", encoding="utf-8") as f:
        f.write("Time\tFiltered_Annual\tFitted_Annual\tResidual\tAnomaly\n")
        for i in range(len(times)):
            t_str = _from_matlab_datenum(float(times[i])).strftime("%Y%m%d")
            above_thr = abs(err[i]) > thr_best
            anomaly = 1 if (above_thr and not in_interf[i]) else 0
            f.write(f"{t_str}\t{high[i]:.3f}\t{fitted_annual[i]:.3f}\t{err[i]:.3f}\t{anomaly}\n")
    print(f"处理序列已写入: {processed_path}")

    # 11. 输出预报效能结果到文本文件
    treating_processes = data_file.stem
    jy = "通过检验（R > R0）" if (Rmax > R0 and Rmax > 0) else "未通过检验（R < R0）"
    nabn = len(ab_st_best)
    out_path = data_file.with_name(data_file.stem + "_result.txt")
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# 破年变异常提取及预报效能结果（单测项）\n")
        f.write(f"# 观测数据文件: {data_file.name}\n")
        f.write(f"# 地震目录文件: {eq_catalog_file.name}\n")
        f.write(f"# 地震筛选规则文件: {polygon_file.name}\n")
        if interference_file is not None:
            f.write(f"# 干扰信息文件: {interference_file.name}\n")
        else:
            f.write("# 干扰信息文件: 无\n")
        f.write(f"# 台站经纬度: {station_lon:.3f}, {station_lat:.3f}\n")
        f.write("#---------------------------------------------\n")
        f.write(f"#------------------------------------ {treating_processes} ({jy}) -----------------------------------\n")
        f.write("# 异常次数   成功预报地震数   漏报地震数    R    R0   最优预报有效期   最优阈值(×STD)  震级下限  震级上限\n")
        f.write(f"{nabn:6d} {success:16d} {miss:16d} {Rmax:8.4f} {R0:8.4f} {best_alm_day:4d} {best_rate:18.2f} {mag_min:8.1f} {mag_max:8.1f}\n")
        f.write("# 具体如下：\n")
        f.write("# 异常开始时间    异常最后时间    是否报准地震   异常距地震发生时间(天)  地震发生时间  地震经度    地震纬度    地震震级    震中距(Km)\n")
        eq_floor = np.floor(eq_times).astype(int)
        for i in range(nabn):
            st = float(ab_st_best[i])
            ed = float(ab_ed_best_from_grid[i])  # 段终点（异常段最后一天）
            win_ed = st + best_alm_day if (ed - st) <= best_alm_day else ed
            st_str = _from_matlab_datenum(st).strftime("%Y%m%d")
            ed_str = _from_matlab_datenum(ed).strftime("%Y%m%d")  # 异常最后时间 = 段终点
            st_int = int(np.ceil(st))
            ed_int = int(np.floor(win_ed))
            eq_in_window = (eq_floor >= st_int) & (eq_floor <= ed_int)
            if not np.any(eq_in_window):
                f.write(f"{st_str}  {ed_str}      否\n")
                continue
            for idx in np.where(eq_in_window)[0]:
                t_eq = float(eq_times[idx])
                days_st = int(np.floor(t_eq - st))
                days_ed = int(np.floor(t_eq - ed))
                eq_date_str = _from_matlab_datenum(t_eq).strftime("%Y%m%d")
                dist_km = _distance_km(eq_lons[idx], eq_lats[idx], station_lon, station_lat)
                f.write(f"{st_str}  {ed_str}      是    {days_st:d}/{days_ed:d}  {eq_date_str}  {eq_lons[idx]:14.2f}  {eq_lats[idx]:5.2f}  {eq_mags[idx]:5.1f}  {dist_km:5.1f}\n")

    # 12. 绘图
    png_path = data_file.with_name(data_file.stem + "_result.png")
    try:
        plot_result_figure(
            times=times,
            series=high_orig,
            low_freq=(high - err),
            err=err,
            eq_times=eq_times,
            eq_lons=eq_lons,
            eq_lats=eq_lats,
            eq_flags=flags,
            poly_x=poly_x,
            poly_y=poly_y,
            station_lon=station_lon,
            station_lat=station_lat,
            rate_series=rate_series,
            alam_days=alam_days,
            R=R,
            best_rate_idx=rate_idx,
            best_alarm_idx=day_idx,
            ab_st_best=ab_st_best,
            ab_ed_best=ab_ed_best_from_grid,
            best_alarm_day=best_alm_day,
            mag_start=mag_min,
            Rmax=Rmax,
            R0=R0,
            out_png=png_path,
            use_cartopy_map=use_cartopy_map,
        )
        print(f"图件已保存: {png_path}")
    except Exception as e:  # 绘图失败不影响数值结果
        print(f"绘图时出现问题（不影响数值结果）: {e}")

    print("计算完成，关键结果如下：")
    print(f"  Rmax = {Rmax:.4f}")
    print(f"  R0   = {R0:.4f}")
    print(f"  最优阈值倍数 = {best_rate:.2f} × STD")
    print(f"  最优预报有效期 = {best_alm_day:d} 天")
    print(f"  成功报准 / 总地震数 = {success} / {len(eq_times)}")
    print(f"结果已写入文件: {out_path}")


if __name__ == "__main__":
    """
    默认示例入口：
    - 请在 README_python.md 中查看如何设置下面这些路径参数；
    - 这里给出一个模板，你只需要把路径改为你本机的实际文件即可。
    """
    # ========= 请按实际情况修改以下参数 =========
    DATA_FILE = Path(r"D:\numerical\cycle-related-anomaly\32016_3_2221.txt")        # 测项观测数据文件
    POLYGON_FILE = Path(r"D:\numerical\cycle-related-anomaly\32016_3_2221_EQ_SelectRule.txt")    # 震例对应规则（多边形，经纬度两列）
    EQ_CATALOG_FILE = Path(r"D:\numerical\cycle-related-anomaly\china3msN.eqt")  # 地震目录
    INTERFERENCE_FILE = Path(r"D:\numerical\cycle-related-anomaly\32016_3_2221_Interference_Period.txt")   # 如有干扰信息表（前两列为起止时间码），在此填写 Path
    STATION_LON = 120.73                       # 台站经度
    STATION_LAT = 31.64                        # 台站纬度
    MAG_MIN = 3.5                              # 震级下限（地震目录筛选）
    MAG_MAX = 9.0                              # 震级上限
    RATE_MIN = 0.0                             # 阈值倍数最小值
    RATE_MAX = 10.0                            # 阈值倍数最大值
    RATE_STEP = 0.1                            # 阈值倍数步长
    ALM_DAY_START = 60                         # 预报期起点（天）
    ALM_DAY_END = 720                          # 预报期终点（天）
    ALM_DAY_STEP = 1                           # 预报期步长（天）
    USE_CARTOPY_MAP = False                    # 右上角空间分布图是否使用 cartopy 底图（默认关闭，无需安装 cartopy）
    # ==============================================

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"DATA_FILE 未找到，请修改为实际路径: {DATA_FILE}")
    if not POLYGON_FILE.exists():
        raise FileNotFoundError(f"POLYGON_FILE 未找到，请修改为实际路径: {POLYGON_FILE}")
    if not EQ_CATALOG_FILE.exists():
        raise FileNotFoundError(f"EQ_CATALOG_FILE 未找到，请修改为实际路径: {EQ_CATALOG_FILE}")

    run_single_station(
        data_file=DATA_FILE,
        polygon_file=POLYGON_FILE,
        eq_catalog_file=EQ_CATALOG_FILE,
        station_lon=STATION_LON,
        station_lat=STATION_LAT,
        interference_file=INTERFERENCE_FILE,
        mag_min=MAG_MIN,
        mag_max=MAG_MAX,
        rate_min=RATE_MIN,
        rate_max=RATE_MAX,
        rate_step=RATE_STEP,
        alm_day_start=ALM_DAY_START,
        alm_day_end=ALM_DAY_END,
        alm_day_step=ALM_DAY_STEP,
        use_cartopy_map=USE_CARTOPY_MAP,
    )