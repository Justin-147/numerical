#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LXX 解算 .NEU → CENC 样式 *_raw.neu 格式转换。

参考模板：`GNSS-coordinated-anomaly/DataIn/SCTQ_raw.neu`

输出格式：
  第 1 行：#Reference position    <lon>     <lat>    <h>    <SITE>
  第 2 行：# YYYYMMDD YYYY.DECM     N(mm)     E(mm)     U(mm) sig_n(mm) sig_e(mm) sig_u(mm)
  后续：按天一行

输入（LXXdata/*.NEU）每行 9 列：
  1 YYYY.DECM
  2 N (m)   → mm
  3 E (m)   → mm
  4 U (m)   → mm
  5 sig_n (m) → mm
  6 sig_e (m) → mm
  7 sig_u (m) → mm
  8 年份 (YYYY)
  9 年内日序号 (DOY，001..366)

站点经纬度：从 `LXXdata/gps_station.dat` 获取（3 列：lon lat name），按站名匹配；
高程固定为 0。
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class StationLL:
    lon: float
    lat: float


# ----------------------------------------------------------------------
# 默认输出时间范围（不想每次命令行传参就改这里）
# - 设为整数 YYYYMMDD 表示截取范围
# - 设为 None 表示不限制（全量输出）
# ----------------------------------------------------------------------
DEFAULT_START_DATE: Optional[int] = 20110101
DEFAULT_END_DATE: Optional[int] = 20161231


def is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def yyyy_decm_noon_from_year_doy(year: int, doy: int) -> float:
    """
    输出与 SCTQ_raw.neu 一致的 YYYY.DECM 定义：该日“中午 12 点”对应的十进制年。
    例如：2011-01-01 12:00 -> 2011 + 0.5/365 = 2011.001369... -> 2011.0014
    """
    days = 366 if is_leap_year(year) else 365
    return float(year) + ((float(doy) - 1.0) + 0.5) / float(days)


def read_station_ll(gps_station_dat: str) -> Dict[str, StationLL]:
    """
    读取 gps_station.dat：每行 3 列（lon lat name）。
    返回 {NAME: StationLL}，NAME 为原样+大写双索引（便于大小写不敏感匹配）。
    """
    out: Dict[str, StationLL] = {}
    with open(gps_station_dat, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                lon = float(parts[0])
                lat = float(parts[1])
            except ValueError:
                continue
            name = parts[2].strip()
            if not name:
                continue
            out[name] = StationLL(lon=lon, lat=lat)
            out[name.upper()] = StationLL(lon=lon, lat=lat)
    return out


def yyyymmdd_from_year_doy(year: int, doy: int) -> int:
    """year + DOY(1-based) → YYYYMMDD（整数）。"""
    dt = date(year, 1, 1) + timedelta(days=doy - 1)
    return int(dt.strftime("%Y%m%d"))


def iter_neu_rows(neu_path: str) -> Iterable[Tuple[float, float, float, float, float, float, float, int, int]]:
    """
    逐行解析 LXX .NEU：返回 (decy, n, e, u, sn, se, su, year, doy)
    """
    with open(neu_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                decy = float(parts[0])
                n_m = float(parts[1])
                e_m = float(parts[2])
                u_m = float(parts[3])
                sn_m = float(parts[4])
                se_m = float(parts[5])
                su_m = float(parts[6])
                year = int(parts[7])
                doy = int(parts[8])
            except ValueError:
                continue
            yield decy, n_m, e_m, u_m, sn_m, se_m, su_m, year, doy


def convert_one_file(
    neu_in: str,
    neu_out: str,
    site: str,
    station_ll: StationLL,
    height_m: float = 0.0,
    start_yyyymmdd: Optional[int] = None,
    end_yyyymmdd: Optional[int] = None,
) -> int:
    """
    转换单个文件，返回写入数据行数。
    """
    os.makedirs(os.path.dirname(neu_out), exist_ok=True)

    header1 = f"#Reference position    {station_ll.lon:.2f}     {station_ll.lat:.2f}    {height_m:.2f}    {site}\n"
    header2 = "# YYYYMMDD YYYY.DECM     N(mm)     E(mm)     U(mm) sig_n(mm) sig_e(mm) sig_u(mm)\n"

    n_written = 0
    with open(neu_out, "w", encoding="utf-8", newline="\n") as out:
        out.write(header1)
        out.write(header2)
        for decy, n_m, e_m, u_m, sn_m, se_m, su_m, year, doy in iter_neu_rows(neu_in):
            ymd = yyyymmdd_from_year_doy(year, doy)
            if start_yyyymmdd is not None and ymd < start_yyyymmdd:
                continue
            if end_yyyymmdd is not None and ymd > end_yyyymmdd:
                continue
            # 时间系统对齐：按输出日期（该日中午 12 点）重新计算 YYYY.DECM（区分闰年）
            decy_out = yyyy_decm_noon_from_year_doy(year, doy)
            n_mm = n_m * 1000.0
            e_mm = e_m * 1000.0
            u_mm = u_m * 1000.0
            sn_mm = sn_m * 1000.0
            se_mm = se_m * 1000.0
            su_mm = su_m * 1000.0

            out.write(
                f"{ymd:9d} {decy_out:10.4f} {n_mm:9.1f} {e_mm:9.1f} {u_mm:9.1f}"
                f" {sn_mm:8.1f} {se_mm:8.1f} {su_mm:9.1f}\n"
            )
            n_written += 1
    return n_written


def main() -> None:
    ap = argparse.ArgumentParser(description="LXX .NEU → *_raw.neu 格式转换")
    ap.add_argument("--lxx-dir", default="LXXdata", help="输入目录（含 .NEU 与 gps_station.dat）")
    ap.add_argument("--out-dir", default=os.path.join("GNSS-coordinated-anomaly", "DataIn"), help="输出目录")
    ap.add_argument("--stations", default=None, help="仅转换指定站点（逗号分隔，如 LS01,LS02）")
    ap.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help="输出起始日期（YYYYMMDD），或 none 表示不限制。默认取脚本内 DEFAULT_START_DATE",
    )
    ap.add_argument(
        "--end-date",
        default=DEFAULT_END_DATE,
        help="输出终止日期（YYYYMMDD），或 none 表示不限制。默认取脚本内 DEFAULT_END_DATE",
    )
    args = ap.parse_args()

    lxx_dir = os.path.abspath(args.lxx_dir)
    out_dir = os.path.abspath(args.out_dir)

    gps_station_dat = os.path.join(lxx_dir, "gps_station.dat")
    if not os.path.isfile(gps_station_dat):
        raise FileNotFoundError(f"未找到 {gps_station_dat}")

    stll = read_station_ll(gps_station_dat)

    wanted = None
    if args.stations:
        wanted = {s.strip() for s in args.stations.split(",") if s.strip()}
        wanted |= {s.upper() for s in wanted}

    neu_files: List[str] = []
    for name in os.listdir(lxx_dir):
        if name.lower().endswith(".neu"):
            if wanted is not None:
                stem = os.path.splitext(name)[0]
                if stem not in wanted and stem.upper() not in wanted:
                    continue
            neu_files.append(os.path.join(lxx_dir, name))
    neu_files.sort()

    if not neu_files:
        raise FileNotFoundError(f"{lxx_dir} 下未找到 .NEU 文件")

    start_ymd = None
    if args.start_date is not None and str(args.start_date).lower() != "none":
        start_ymd = int(str(args.start_date))
    end_ymd = None
    if args.end_date is not None and str(args.end_date).lower() != "none":
        end_ymd = int(str(args.end_date))
    if start_ymd is not None and end_ymd is not None and start_ymd > end_ymd:
        raise ValueError(f"start-date({start_ymd}) 不能晚于 end-date({end_ymd})")

    ok = 0
    for neu_in in neu_files:
        stem = os.path.splitext(os.path.basename(neu_in))[0]
        if stem in stll:
            ll = stll[stem]
        elif stem.upper() in stll:
            ll = stll[stem.upper()]
        else:
            raise KeyError(f"gps_station.dat 中找不到站点 {stem} 的经纬度")

        neu_out = os.path.join(out_dir, f"{stem}_raw.neu")
        n = convert_one_file(
            neu_in,
            neu_out,
            stem,
            ll,
            height_m=0.0,
            start_yyyymmdd=start_ymd,
            end_yyyymmdd=end_ymd,
        )
        print(f"OK {os.path.basename(neu_in)} -> {os.path.basename(neu_out)} ({n} lines written)")
        ok += 1

    print(f"完成：{ok} 个文件输出到 {out_dir}")


if __name__ == "__main__":
    main()

