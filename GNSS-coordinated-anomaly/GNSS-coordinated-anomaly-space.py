#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNSS 协调方向异常：空间格网统计 + 绘图

输入（来自 GNSS-coordinated-anomaly-filt.py 的输出）：
- FiltDataOut/stinfo.txt
- FiltDataOut/*_HHTfilt.txt 或 *_Bandfilt.txt（与 filt 的 --filter-mode 一致）

支持按时间范围或指定日期计算/出图：
- --date-start YYYYMMDD --date-end YYYYMMDD
- --dates YYYYMMDD,YYYYMMDD,...

说明：
- 若未指定日期参数，则使用站点文件中的完整日期序列。
- 站点文件为逐日连续序列（缺测已插值），满足等间隔采样前提。
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np


DEFAULT_OUT_PATH = os.path.join("GNSS-coordinated-anomaly", "FiltDataOut")
DEFAULT_FRAMES_OUT = os.path.join("GNSS-coordinated-anomaly", "Frames")
# 不带参数运行时的默认行为：完整执行 grid + plot
DEFAULT_RUN_WHEN_NO_ARGS = "pipeline"  # "grid" | "plot" | "pipeline"

# 生成模式：all 先 hht 后 bandpass；hht/bandpass 则只生成对应一套
DEFAULT_GEN_MODE = "all"  # "all" | "hht" | "bandpass"

# 默认时间选择（命令行不传时生效）
# - 若 DEFAULT_DATES 非空：优先按指定日期点计算/出图
# - 否则按 [DEFAULT_DATE_START, DEFAULT_DATE_END] 的范围
DEFAULT_DATE_START = 20130201
DEFAULT_DATE_END = 20130430
DEFAULT_DATES = None # [20130224, 20130330, 20130405, 20130413, 20130418]

# ---------------------------------------------------------------------------
# 空间范围默认参数
# - DEFAULT_LAT/LON_* 为 None 表示：自动用 stinfo 的站点范围（并加 margin）
# ---------------------------------------------------------------------------
DEFAULT_LAT_MIN = None
DEFAULT_LAT_MAX = None
DEFAULT_LON_MIN = None
DEFAULT_LON_MAX = None
DEFAULT_BOUNDS_MARGIN_DEG = 0.5

# 角差平均的绘图范围（度）
DEFAULT_ANGDIFF_VMIN = 0.0
DEFAULT_ANGDIFF_VMAX = 120.0

# ---------------------------------------------------------------------------
# 格网与邻域参数（默认值）
# ---------------------------------------------------------------------------
DEFAULT_GRID_STEP_DEG = 0.1
DEFAULT_SEARCH_RADIUS_KM = 50.0
DEFAULT_MIN_STATIONS = 3


def _angdiff_cmap():
    # 参考色标（值, R, G, B）：
    # 0   255 0   0
    # 20  255 255 0
    # 40  71  254 218
    # 60  0   213 254
    # 80  0   101 254
    # 100 0   0   242
    # 120 0   0   137
    from matplotlib.colors import LinearSegmentedColormap

    vmax = float(DEFAULT_ANGDIFF_VMAX)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 120.0

    anchors = [
        (0.0, (255, 0, 0)),
        (20.0, (255, 255, 0)),
        (40.0, (71, 254, 218)),
        (60.0, (0, 213, 254)),
        (80.0, (0, 101, 254)),
        (100.0, (0, 0, 242)),
        (120.0, (0, 0, 137)),
    ]
    stops = []
    for v, (r, g, b) in anchors:
        p = max(0.0, min(1.0, float(v) / vmax))
        stops.append((p, (float(r) / 255.0, float(g) / 255.0, float(b) / 255.0)))
    return LinearSegmentedColormap.from_list("angdiff_ref", stops)


def _list_station_filt_txt_files(out_path: str, tag: str) -> List[str]:
    return sorted(glob.glob(os.path.join(out_path, f"*_{tag}.txt")))


def _station_filt_txt_path(out_path: str, site_id: str, tag: str) -> Optional[str]:
    p = os.path.join(out_path, f"{site_id}_{tag}.txt")
    return p if os.path.isfile(p) else None


def _mode_to_tag(mode: str) -> str:
    m = str(mode).strip().lower()
    if m == "hht":
        return "HHTfilt"
    if m == "bandpass":
        return "Bandfilt"
    raise ValueError(f"未知模式：{mode}（应为 hht/bandpass）")


def load_stinfo(out_path: str) -> np.ndarray:
    p = os.path.join(out_path, "stinfo.txt")
    rows: List[Tuple[float, float, str]] = []
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("site"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            site = parts[0]
            lat = float(parts[1])
            lon = float(parts[2])
            rows.append((lat, lon, site))
    out = np.empty((len(rows), 3), dtype=object)
    for i, (lat, lon, site) in enumerate(rows):
        out[i, 0] = lat
        out[i, 1] = lon
        out[i, 2] = site
    return out


def load_station_hhtfilt(path: str) -> Dict[str, np.ndarray]:
    """
    读取 <site>_HHTfilt.txt 或 <site>_Bandfilt.txt
    列：YYYYMMDD, YYYY.DECM, N/E/U_filt(mm), Azimuth(deg)
    返回：days(int), decm(float), NS/EW/ZZ(m), AA(deg)
    """
    data = np.loadtxt(path, delimiter="\t", comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 6:
        raise ValueError(f"站点文件列数不足（期望>=6）：{path}")
    days = data[:, 0].astype(np.int32)
    decm = data[:, 1].astype(np.float64)
    ns_m = (data[:, 2] / 1000.0).astype(np.float64)
    ew_m = (data[:, 3] / 1000.0).astype(np.float64)
    zz_m = (data[:, 4] / 1000.0).astype(np.float64)
    aa = data[:, 5].astype(np.float64)
    return {"days": days, "decm": decm, "NS_ll": ns_m, "EW_ll": ew_m, "ZZ_ll": zz_m, "AA": aa}


def _take_series_by_days(days: np.ndarray, values: np.ndarray, sel_days: np.ndarray) -> np.ndarray:
    """
    将某站点的逐日序列按 sel_days 对齐取值。
    若某天不存在则填 NaN（避免不同站点长度/起止日期不一致导致越界）。
    """
    days = np.asarray(days, dtype=np.int64).ravel()
    values = np.asarray(values, dtype=np.float64).ravel()
    sel_days = np.asarray(sel_days, dtype=np.int64).ravel()
    out = np.full(sel_days.size, np.nan, dtype=np.float64)
    if days.size == 0 or values.size == 0 or sel_days.size == 0:
        return out
    # days 必须升序（本项目写出的逐日序列满足）
    idx = np.searchsorted(days, sel_days)
    ok = (idx >= 0) & (idx < days.size)
    if np.any(ok):
        idx2 = idx[ok]
        ok2 = days[idx2] == sel_days[ok]
        if np.any(ok2):
            out_idx = np.flatnonzero(ok)[ok2]
            out[out_idx] = values[idx2[ok2]]
    return out


def angular_diff_deg(a1: float, a2: float) -> float:
    dd = abs(a1 - a2) % 360.0
    return dd if dd <= 180.0 else 360.0 - dd


def _approx_distance_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    近似球面距离（km），用于小范围快速筛选。
    """
    lat1r = np.deg2rad(lat1)
    lon1r = np.deg2rad(lon1)
    lat2r = np.deg2rad(lat2.astype(np.float64))
    lon2r = np.deg2rad(lon2.astype(np.float64))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    # equirectangular approximation
    x = dlon * np.cos((lat1r + lat2r) / 2.0)
    y = dlat
    return 6371.0 * np.sqrt(x * x + y * y)


def _resolve_bounds_from_stations(
    lo: np.ndarray,
    *,
    lat_min: Optional[float],
    lat_max: Optional[float],
    lon_min: Optional[float],
    lon_max: Optional[float],
    margin_deg: float,
) -> Tuple[float, float, float, float]:
    """
    若用户未显式指定经纬度范围，则从 stinfo 的站点经纬度范围自动推断。
    """
    lat_s = lo[:, 0].astype(float)
    lon_s = lo[:, 1].astype(float)
    if lat_min is None:
        lat_min = float(np.nanmin(lat_s)) - float(margin_deg)
    if lat_max is None:
        lat_max = float(np.nanmax(lat_s)) + float(margin_deg)
    if lon_min is None:
        lon_min = float(np.nanmin(lon_s)) - float(margin_deg)
    if lon_max is None:
        lon_max = float(np.nanmax(lon_s)) + float(margin_deg)
    if lat_min >= lat_max or lon_min >= lon_max:
        raise ValueError("自动推断经纬度范围失败：请手动指定 --lat-min/max 与 --lon-min/max")
    return float(lat_min), float(lat_max), float(lon_min), float(lon_max)


def _parse_dates_list(s: str) -> List[int]:
    # 允许两种输入：
    # 1) "20130415,20130418"
    # 2) "[20130415, 20130418]"（用户习惯的列表形式）
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    out: List[int] = []
    for t in s.split(","):
        t = t.strip()
        if not t:
            continue
        if len(t) != 8 or not t.isdigit():
            raise ValueError(f"日期必须为 YYYYMMDD：{t}")
        out.append(int(t))
    if not out:
        raise ValueError("dates 为空")
    return out


def _select_day_indices(ref_days: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    ref_days = np.asarray(ref_days, dtype=np.int32)
    dates = args.dates
    if dates is None and DEFAULT_DATES:
        dates = ",".join(str(int(x)) for x in DEFAULT_DATES)
    if dates is not None:
        want = set(_parse_dates_list(str(dates)))
        idx = np.flatnonzero(np.isin(ref_days.astype(np.int64), np.array(sorted(want), dtype=np.int64)))
        return idx.astype(np.int32)
    ds = args.date_start if args.date_start is not None else DEFAULT_DATE_START
    de = args.date_end if args.date_end is not None else DEFAULT_DATE_END
    if ds is not None or de is not None:
        lo = int(ds) if ds is not None else int(ref_days.min())
        hi = int(de) if de is not None else int(ref_days.max())
        if lo > hi:
            raise ValueError("date-start 不能大于 date-end")
        idx = np.flatnonzero((ref_days >= lo) & (ref_days <= hi))
        return idx.astype(np.int32)
    return np.arange(ref_days.size, dtype=np.int32)


def _run_grid_one(args: argparse.Namespace, mode: str) -> None:
    lo = load_stinfo(args.out_path)
    tag = _mode_to_tag(mode)
    cand = _list_station_filt_txt_files(args.out_path, tag)
    if not cand:
        raise FileNotFoundError(f"未找到 *_{tag}.txt，请先运行 filt（mode={mode}）")
    ref = load_station_hhtfilt(cand[0])
    ref_days = ref["days"]
    sel_idx = _select_day_indices(ref_days, args)
    if sel_idx.size == 0:
        raise ValueError("选择的日期为空（可能超出站点文件日期范围）")
    sel_days = ref_days[sel_idx]
    doy = int(sel_idx.size)

    lat_min, lat_max, lon_min, lon_max = _resolve_bounds_from_stations(
        lo,
        lat_min=getattr(args, "lat_min", None),
        lat_max=getattr(args, "lat_max", None),
        lon_min=getattr(args, "lon_min", None),
        lon_max=getattr(args, "lon_max", None),
        margin_deg=float(getattr(args, "bounds_margin_deg", DEFAULT_BOUNDS_MARGIN_DEG)),
    )

    intv = float(args.grid_step)
    radius_km = float(args.search_radius_km)
    min_stations = int(args.min_stations)
    lats = np.arange(lat_min, lat_max + intv * 0.5, intv)
    lons = np.arange(lon_min, lon_max + intv * 0.5, intv)
    n_lat, n_lon = len(lats), len(lons)

    angdiff_mean = np.full((n_lat, n_lon, doy), np.nan)  # 平均角差（度）
    st_lat = lo[:, 0].astype(np.float64)
    st_lon = lo[:, 1].astype(np.float64)

    for li, lat0 in enumerate(lats):
        for oi, lon0 in enumerate(lons):
            dist_km = _approx_distance_km(float(lat0), float(lon0), st_lat, st_lon)
            st = np.where(dist_km <= radius_km)[0]
            if len(st) < min_stations:
                continue
            ang_list: List[np.ndarray] = []
            for idx in st:
                sid = str(lo[idx, 2])
                fp = _station_filt_txt_path(args.out_path, sid, tag)
                if not fp:
                    continue
                z = load_station_hhtfilt(fp)
                ew_ll = _take_series_by_days(z["days"], z["EW_ll"], sel_days)
                ns_ll = _take_series_by_days(z["days"], z["NS_ll"], sel_days)
                aa = np.degrees(np.arctan2(ew_ll, ns_ll))
                aa = np.where(aa < 0, aa + 360.0, aa)
                ang_list.append(aa)
            if not ang_list:
                continue
            ang = np.column_stack(ang_list)
            analen = ang.shape[1]
            for day in range(doy):
                dds = 0.0
                stnn = 0
                for j1 in range(analen - 1):
                    for j2 in range(j1 + 1, analen):
                        a1, a2 = ang[day, j1], ang[day, j2]
                        if np.isfinite(a1) and np.isfinite(a2) and a1 >= 0 and a2 >= 0:
                            stnn += 1
                            dds += angular_diff_deg(float(a1), float(a2))
                if stnn > 0:
                    angdiff_mean[li, oi, day] = float(dds / stnn)

    os.makedirs(args.frames_out, exist_ok=True)
    out_ad = os.path.join(args.frames_out, f"ANGDIFF_MEAN_{str(mode).strip().lower()}.txt")
    # 写出每个格点每一天的角差平均：YYYYMMDD lon lat angdiff_deg
    # 只写非 NaN 行，避免文件过大
    with open(out_ad, "w", encoding="utf-8", newline="\n") as f:
        f.write("YYYYMMDD\tlon\tlat\tanglediff_mean(deg)\n")
        for di in range(doy):
            a = angdiff_mean[:, :, di]
            ii, jj = np.where(np.isfinite(a))
            for i0, j0 in zip(ii.tolist(), jj.tolist()):
                f.write(
                    f"{int(sel_days[di])}\t{float(lons[j0]):.6f}\t{float(lats[i0]):.6f}\t{float(a[i0, j0]):.10g}\n"
                )
    print(f"格网完成：mode={mode}, doy={doy}（选中日期数），ANGDIFF_MEAN 已写入 {out_ad}")


def run_grid(args: argparse.Namespace) -> None:
    m = str(getattr(args, "gen_mode", "all")).strip().lower()
    modes = ["hht", "bandpass"] if m == "all" else [m]
    for mm in modes:
        _run_grid_one(args, mm)


def _run_plot_one(args: argparse.Namespace, mode: str) -> None:
    # 目前 plot 复用 grid 的计算（按选中日期子集），逐日输出 PNG
    os.makedirs(args.frames_out, exist_ok=True)

    # 先拿参考日期序列与索引
    tag = _mode_to_tag(mode)
    cand = _list_station_filt_txt_files(args.out_path, tag)
    if not cand:
        raise FileNotFoundError(f"未找到 *_{tag}.txt，请先运行 filt（mode={mode}）")
    ref = load_station_hhtfilt(cand[0])
    ref_days = ref["days"]
    sel_idx = _select_day_indices(ref_days, args)
    if sel_idx.size == 0:
        raise ValueError("选择的日期为空")
    sel_days = ref_days[sel_idx]

    # 这里直接在 plot 内部计算一次格网（按选中日期子集），避免引入额外状态文件。
    #
    # 若你后续希望进一步加速，我们可以改成“单日即时计算格网 + 出图”避免 3D 常驻内存。
    lo = load_stinfo(args.out_path)
    lat_min, lat_max, lon_min, lon_max = _resolve_bounds_from_stations(
        lo,
        lat_min=getattr(args, "lat_min", None),
        lat_max=getattr(args, "lat_max", None),
        lon_min=getattr(args, "lon_min", None),
        lon_max=getattr(args, "lon_max", None),
        margin_deg=float(getattr(args, "bounds_margin_deg", DEFAULT_BOUNDS_MARGIN_DEG)),
    )
    intv = float(args.grid_step)
    radius_km = float(args.search_radius_km)
    min_stations = int(args.min_stations)
    lats = np.arange(lat_min, lat_max + intv * 0.5, intv)
    lons = np.arange(lon_min, lon_max + intv * 0.5, intv)
    n_lat, n_lon = len(lats), len(lons)
    doy = int(sel_idx.size)

    angdiff_mean = np.full((n_lat, n_lon, doy), np.nan)
    st_lat = lo[:, 0].astype(np.float64)
    st_lon = lo[:, 1].astype(np.float64)

    for li, lat0 in enumerate(lats):
        for oi, lon0 in enumerate(lons):
            dist_km = _approx_distance_km(float(lat0), float(lon0), st_lat, st_lon)
            st = np.where(dist_km <= radius_km)[0]
            if len(st) < min_stations:
                continue
            ang_list: List[np.ndarray] = []
            for idx in st:
                sid = str(lo[idx, 2])
                fp = _station_filt_txt_path(args.out_path, sid, tag)
                if not fp:
                    continue
                z = load_station_hhtfilt(fp)
                ew_ll = _take_series_by_days(z["days"], z["EW_ll"], sel_days)
                ns_ll = _take_series_by_days(z["days"], z["NS_ll"], sel_days)
                aa = np.degrees(np.arctan2(ew_ll, ns_ll))
                aa = np.where(aa < 0, aa + 360.0, aa)
                ang_list.append(aa)
            if not ang_list:
                continue
            ang = np.column_stack(ang_list)
            analen = ang.shape[1]
            for day in range(doy):
                dds = 0.0
                stnn = 0
                for j1 in range(analen - 1):
                    for j2 in range(j1 + 1, analen):
                        a1, a2 = ang[day, j1], ang[day, j2]
                        if np.isfinite(a1) and np.isfinite(a2) and a1 >= 0 and a2 >= 0:
                            stnn += 1
                            dds += angular_diff_deg(float(a1), float(a2))
                if stnn > 0:
                    angdiff_mean[li, oi, day] = float(dds / stnn)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _plot_gps_index_frame(
        out_png: str,
        angdiff_day: np.ndarray,
        lo_st: np.ndarray,
        st_u: np.ndarray,
        st_v: np.ndarray,
        lats_: np.ndarray,
        lons_: np.ndarray,
        day_label: str,
        *,
        station_arrow_scale: float,
        vmin: float,
        vmax: float,
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 7))
        cmap = _angdiff_cmap()
        im = ax.imshow(
            angdiff_day,
            origin="lower",
            aspect="equal",
            extent=[float(lons_[0]), float(lons_[-1]), float(lats_[0]), float(lats_[-1])],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(im, ax=ax, shrink=0.65, fraction=0.03, pad=0.02, label="mean angular diff (deg)")
        ax.set_title(f"Mean Angle Difference of GNSS Observation Vectors ({day_label})")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")

        # 叠加箭头：在站点实际位置画 N/E（m -> deg 位移，并根据图框范围自动缩放）
        # 目标：5mm 的箭头长度约占图框尺度的 1/15（再乘 station_arrow_scale 作为额外倍率）
        lat_span = float(lats_[-1]) - float(lats_[0])
        lon_span = float(lons_[-1]) - float(lons_[0])
        desired_len_deg = max(1e-12, min(lat_span, lon_span) / 15.0)
        ref_m = 0.005  # 5mm
        ref_deg = ref_m / 110574.0  # 近似：1deg lat ≈ 110.574 km
        auto_fac = desired_len_deg / max(1e-18, ref_deg)
        fac = float(station_arrow_scale) * auto_fac

        st_lat_deg = lo_st[:, 0].astype(np.float64)
        m_per_deg_lon = 111320.0 * np.cos(np.deg2rad(st_lat_deg))
        m_per_deg_lon = np.maximum(m_per_deg_lon, 1e-6)
        u_deg = (st_u.astype(np.float64) / m_per_deg_lon) * fac  # East -> dlon(deg)
        v_deg = (st_v.astype(np.float64) / 110574.0) * fac  # North -> dlat(deg)

        q = ax.quiver(
            lo_st[:, 1].astype(float),
            lo_st[:, 0].astype(float),
            u_deg,
            v_deg,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            pivot="tail",
            width=0.003,
            alpha=0.9,
            color="k",
        )

        # 左下角比例尺（5mm，换算到当前绘图单位：deg 位移）
        ref_key = (ref_m / 110574.0) * fac
        ax.quiverkey(q, 0.12, 0.10, ref_key, "5 mm", labelpos="E", coordinates="axes")
        fig.tight_layout()
        fig.savefig(out_png, dpi=160)
        plt.close(fig)

    for i, ymd in enumerate(sel_days):
        day_label = str(int(ymd))
        if args.make_gps_index:
            out_png = os.path.join(args.frames_out, f"GNSS-coordinated-anomaly-{str(mode).strip().lower()}-{day_label}.png")
            # 站点箭头：直接取每站该天的 (E,N) 滤波值
            st_lats = []
            st_lons = []
            st_u = []  # East
            st_v = []  # North
            for row in lo:
                sid = str(row[2])
                fp = _station_filt_txt_path(args.out_path, sid, tag)
                if not fp:
                    continue
                z = load_station_hhtfilt(fp)
                e0 = float(_take_series_by_days(z["days"], z["EW_ll"], np.array([int(ymd)], dtype=np.int64))[0])
                n0 = float(_take_series_by_days(z["days"], z["NS_ll"], np.array([int(ymd)], dtype=np.int64))[0])
                if not (np.isfinite(e0) and np.isfinite(n0)):
                    continue
                st_lats.append(float(row[0]))
                st_lons.append(float(row[1]))
                st_u.append(e0)
                st_v.append(n0)
            _plot_gps_index_frame(
                out_png,
                angdiff_mean[:, :, i],
                np.column_stack([np.array(st_lats), np.array(st_lons)]),
                np.array(st_u, dtype=np.float64),
                np.array(st_v, dtype=np.float64),
                lats,
                lons,
                day_label,
                station_arrow_scale=float(args.station_arrow_scale),
                vmin=float(args.angdiff_vmin),
                vmax=float(args.angdiff_vmax),
            )

    print(f"绘图完成：mode={mode}, 输出到 {args.frames_out}（帧数 {len(sel_days)}）")


def run_plot(args: argparse.Namespace) -> None:
    m = str(getattr(args, "gen_mode", "all")).strip().lower()
    modes = ["hht", "bandpass"] if m == "all" else [m]
    for mm in modes:
        _run_plot_one(args, mm)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GNSS 协调方向异常：空间格网统计与绘图")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_time_args(pp: argparse.ArgumentParser) -> None:
        pp.add_argument(
            "--date-start",
            type=int,
            default=None,
            help=f"起始日期 YYYYMMDD（含）；默认 {DEFAULT_DATE_START}",
        )
        pp.add_argument(
            "--date-end",
            type=int,
            default=None,
            help=f"结束日期 YYYYMMDD（含）；默认 {DEFAULT_DATE_END}",
        )
        pp.add_argument(
            "--dates",
            type=str,
            default=None,
            help=f"指定日期列表（优先级最高）：YYYYMMDD,YYYYMMDD,... 或 [..]；默认 {DEFAULT_DATES}",
        )

    def add_gen_mode_args(pp: argparse.ArgumentParser) -> None:
        pp.add_argument(
            "--gen-mode",
            type=str,
            default=DEFAULT_GEN_MODE,
            choices=["all", "hht", "bandpass"],
            help="生成模式：hht 只用 *_HHTfilt；bandpass 只用 *_Bandfilt；all 依次生成两套（默认 all）",
        )

    pg = sub.add_parser("grid", help="读取 stinfo + *_HHTfilt.txt / *_Bandfilt.txt 计算格网统计")
    pg.add_argument("--out-path", default=DEFAULT_OUT_PATH)
    pg.add_argument("--frames-out", default=DEFAULT_FRAMES_OUT, help="数据/图件输出目录（默认 Frames）")
    add_gen_mode_args(pg)
    add_time_args(pg)
    pg.add_argument("--grid-step", type=float, default=DEFAULT_GRID_STEP_DEG, help="格网步长(度)")
    pg.add_argument("--search-radius-km", type=float, default=DEFAULT_SEARCH_RADIUS_KM, help="搜索半径(km)")
    pg.add_argument("--min-stations", type=int, default=DEFAULT_MIN_STATIONS, help="最小站点数")
    pg.add_argument("--lat-min", type=float, default=DEFAULT_LAT_MIN, help="格网纬度下限；None 表示自动取站点范围")
    pg.add_argument("--lat-max", type=float, default=DEFAULT_LAT_MAX, help="格网纬度上限；None 表示自动取站点范围")
    pg.add_argument("--lon-min", type=float, default=DEFAULT_LON_MIN, help="格网经度下限；None 表示自动取站点范围")
    pg.add_argument("--lon-max", type=float, default=DEFAULT_LON_MAX, help="格网经度上限；None 表示自动取站点范围")
    pg.add_argument("--bounds-margin-deg", type=float, default=DEFAULT_BOUNDS_MARGIN_DEG, help="自动范围时的边界余量(度)")
    pg.add_argument("--anom-lat", type=float, default=44.0)
    pg.add_argument("--anom-lon", type=float, default=142.0)
    pg.add_argument("--smallyc", type=float, default=0.5)
    pg.add_argument("--largeyc", type=float, default=2.0)
    pg.set_defaults(func=run_grid)

    pp = sub.add_parser("plot", help="基于格网统计结果逐日绘图")
    pp.add_argument("--out-path", default=DEFAULT_OUT_PATH)
    pp.add_argument("--frames-out", default=DEFAULT_FRAMES_OUT)
    add_gen_mode_args(pp)
    add_time_args(pp)
    pp.add_argument("--make-gps-index", action="store_true", default=True)
    pp.add_argument("--station-arrow-scale", type=float, default=0.002, help="站点箭头缩放（越小箭头越长；单位按 m）")
    pp.add_argument("--angdiff-vmin", type=float, default=DEFAULT_ANGDIFF_VMIN, help="角差色标最小值(度)")
    pp.add_argument("--angdiff-vmax", type=float, default=DEFAULT_ANGDIFF_VMAX, help="角差色标最大值(度)")
    pp.add_argument("--grid-step", type=float, default=DEFAULT_GRID_STEP_DEG, help="格网步长(度)")
    pp.add_argument("--search-radius-km", type=float, default=DEFAULT_SEARCH_RADIUS_KM, help="搜索半径(km)")
    pp.add_argument("--min-stations", type=int, default=DEFAULT_MIN_STATIONS, help="最小站点数")
    pp.add_argument("--lat-min", type=float, default=DEFAULT_LAT_MIN, help="格网纬度下限；None 表示自动取站点范围")
    pp.add_argument("--lat-max", type=float, default=DEFAULT_LAT_MAX, help="格网纬度上限；None 表示自动取站点范围")
    pp.add_argument("--lon-min", type=float, default=DEFAULT_LON_MIN, help="格网经度下限；None 表示自动取站点范围")
    pp.add_argument("--lon-max", type=float, default=DEFAULT_LON_MAX, help="格网经度上限；None 表示自动取站点范围")
    pp.add_argument("--bounds-margin-deg", type=float, default=DEFAULT_BOUNDS_MARGIN_DEG, help="自动范围时的边界余量(度)")
    pp.add_argument("--anom-lat", type=float, default=44.0)
    pp.add_argument("--anom-lon", type=float, default=142.0)
    pp.add_argument("--smallyc", type=float, default=0.5)
    pp.add_argument("--largeyc", type=float, default=2.0)
    pp.set_defaults(func=run_plot)

    return p


def main() -> None:
    p = build_parser()
    if len(sys.argv) == 1:
        # 允许用户直接运行脚本
        base = dict(
            out_path=DEFAULT_OUT_PATH,
            frames_out=DEFAULT_FRAMES_OUT,
            gen_mode=DEFAULT_GEN_MODE,
            date_start=None,
            date_end=None,
            dates=None,
            grid_step=DEFAULT_GRID_STEP_DEG,
            search_radius_km=DEFAULT_SEARCH_RADIUS_KM,
            min_stations=DEFAULT_MIN_STATIONS,
            lat_min=DEFAULT_LAT_MIN,
            lat_max=DEFAULT_LAT_MAX,
            lon_min=DEFAULT_LON_MIN,
            lon_max=DEFAULT_LON_MAX,
            bounds_margin_deg=DEFAULT_BOUNDS_MARGIN_DEG,
            anom_lat=44.0,
            anom_lon=142.0,
            smallyc=0.5,
            largeyc=2.0,
            make_gps_index=True,
            station_arrow_scale=1.0,
            angdiff_vmin=DEFAULT_ANGDIFF_VMIN,
            angdiff_vmax=DEFAULT_ANGDIFF_VMAX,
        )

        if DEFAULT_RUN_WHEN_NO_ARGS == "grid":
            run_grid(argparse.Namespace(**{k: base[k] for k in base if k not in ("frames_out", "make_gps_index", "station_arrow_scale", "angdiff_vmin", "angdiff_vmax")}))
        elif DEFAULT_RUN_WHEN_NO_ARGS == "plot":
            run_plot(argparse.Namespace(**base))
        else:
            # pipeline: 先算 grid，再出图
            run_grid(argparse.Namespace(**{k: base[k] for k in base if k not in ("make_gps_index", "station_arrow_scale", "angdiff_vmin", "angdiff_vmax")}))
            run_plot(argparse.Namespace(**base))
        return

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

