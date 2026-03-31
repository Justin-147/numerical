#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CrossFault-FaultAnomaly.py
=========================

读取：
1) 跨断层异常表（示例：跨断层异常表-示例.xlsx）
   固定表头：场地名称、手段名称、经度、纬度
2) 断层段迹线坐标（示例：FaultCord_justExample.xlsx）
   固定表头：断层编号、断层段编号、断层名称、断层段名称、经度、纬度

对每个异常场地（点），计算其到每条断层段（折线）的最小距离，
筛选距离阈值（默认 5 km）内且最近的断层段，作为异常断层段。

输出：
Abnormal_Fault_Segments_from_CrossFault.txt
字段（tab 分隔）：
fault_id  segment_id  fault_name  segment_name  abnormal_site  abnormal_item  lon_site  lat_site

说明：
- 距离在局部近似平面（等距矩形投影）下计算，单位 km，适用于“5 km 邻域”筛选。
- 若某异常点 5 km 内没有任何断层段，则不输出该点记录，但仍生成输出文件（仅表头）。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


# ============
# 可修改参数区
# ============

ABNORMAL_XLSX = "跨断层异常表-示例.xlsx" # 跨断层异常表
FAULT_XLSX = "FaultCord_justExample.xlsx" # 断层段迹线坐标
OUTPUT_TXT = "Abnormal_Fault_Segments_from_CrossFault.txt" # 输出文件

# 距离阈值（km）
MAX_DIST_KM = 5.0

# 固定表头
ABNORMAL_COLUMNS = {
    "site": "场地名称",
    "item": "手段名称",
    "lon": "经度",
    "lat": "纬度",
}
FAULT_COLUMNS = {
    "fault_id": "断层编号",
    "segment_id": "断层段编号",
    "fault_name": "断层名称",
    "segment_name": "断层段名称",
    "lon": "经度",
    "lat": "纬度",
}

OUTPUT_HEADER = (
    "#fault_id\tsegment_id\tfault_name\tsegment_name\t"
    "abnormal_site\tabnormal_item\tlon_site\tlat_site\n"
)


def _require_columns(df: pd.DataFrame, cols: dict, name: str) -> None:
    missing = [v for v in cols.values() if v not in df.columns]
    if missing:
        raise ValueError(f"{name} 缺少固定列：{missing}")


def _to_local_xy_m(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 lon/lat（deg）转换到以 (lon0,lat0) 为中心的局部平面坐标（m）。
    """
    # WGS84 近似：1 deg lat ≈ 111.32 km；1 deg lon ≈ 111.32*cos(lat0) km
    k = 111_320.0
    x = (lon - lon0) * k * math.cos(math.radians(lat0))
    y = (lat - lat0) * k
    return x, y


def _point_to_segment_distance_m(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """
    点 P 到线段 (x1,y1)-(x2,y2) 的最小距离（m）。
    """
    vx = x2 - x1
    vy = y2 - y1
    wx = px - x1
    wy = py - y1
    vv = vx * vx + vy * vy
    if vv == 0.0:
        return math.hypot(px - x1, py - y1)
    t = (wx * vx + wy * vy) / vv
    if t <= 0.0:
        return math.hypot(px - x1, py - y1)
    if t >= 1.0:
        return math.hypot(px - x2, py - y2)
    projx = x1 + t * vx
    projy = y1 + t * vy
    return math.hypot(px - projx, py - projy)


def _point_to_polyline_min_distance_km(lon0: float, lat0: float, poly_lonlat: np.ndarray) -> float:
    """
    点 (lon0,lat0) 到折线 poly_lonlat(n×2: lon,lat) 的最小距离（km）。
    """
    if poly_lonlat.shape[0] < 2:
        return float("inf")
    poly_lon = poly_lonlat[:, 0].astype(float)
    poly_lat = poly_lonlat[:, 1].astype(float)
    xs, ys = _to_local_xy_m(poly_lon, poly_lat, lon0, lat0)
    px, py = 0.0, 0.0  # 点作为局部坐标原点
    dmin = float("inf")
    for i in range(len(xs) - 1):
        d = _point_to_segment_distance_m(px, py, float(xs[i]), float(ys[i]), float(xs[i + 1]), float(ys[i + 1]))
        if d < dmin:
            dmin = d
    return dmin / 1000.0


def main() -> None:
    work_dir = Path(__file__).resolve().parent
    abnormal_xlsx = (work_dir / ABNORMAL_XLSX).resolve()
    fault_xlsx = (work_dir / FAULT_XLSX).resolve()
    out_path = (work_dir / OUTPUT_TXT).resolve()

    abn_df = pd.read_excel(abnormal_xlsx, sheet_name=0)
    fault_df = pd.read_excel(fault_xlsx, sheet_name=0)
    abn_df.columns = [str(c).strip() for c in abn_df.columns]
    fault_df.columns = [str(c).strip() for c in fault_df.columns]

    _require_columns(abn_df, ABNORMAL_COLUMNS, "跨断层异常表")
    _require_columns(fault_df, FAULT_COLUMNS, "断层段迹线表")

    # 断层段分组：同一 (fault_id, segment_id) 组成一条折线
    seg_groups = fault_df.groupby([FAULT_COLUMNS["fault_id"], FAULT_COLUMNS["segment_id"]], dropna=False)
    fault_segments: List[Tuple[str, str, str, str, np.ndarray]] = []
    for (fid, sid), g in seg_groups:
        g2 = g.dropna(subset=[FAULT_COLUMNS["lon"], FAULT_COLUMNS["lat"]])
        if g2.shape[0] < 2:
            continue
        coords = g2[[FAULT_COLUMNS["lon"], FAULT_COLUMNS["lat"]]].to_numpy(dtype=float)
        fname = str(g2[FAULT_COLUMNS["fault_name"]].iloc[0])
        sname = str(g2[FAULT_COLUMNS["segment_name"]].iloc[0])
        fault_segments.append((str(fid), str(sid), fname, sname, coords))

    records: List[Tuple[str, str, str, str, str, str, float, float]] = []
    for _, row in abn_df.iterrows():
        site = str(row[ABNORMAL_COLUMNS["site"]]).strip()
        item = str(row[ABNORMAL_COLUMNS["item"]]).strip()
        lon_site = float(row[ABNORMAL_COLUMNS["lon"]])
        lat_site = float(row[ABNORMAL_COLUMNS["lat"]])

        best = None
        best_d = None
        for fid, sid, fname, sname, poly in fault_segments:
            d_km = _point_to_polyline_min_distance_km(lon_site, lat_site, poly)
            if best_d is None or d_km < best_d:
                best_d = d_km
                best = (fid, sid, fname, sname)

        if best is None or best_d is None:
            continue
        if best_d <= MAX_DIST_KM:
            fid, sid, fname, sname = best
            records.append((fid, sid, fname, sname, site, item, lon_site, lat_site))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(OUTPUT_HEADER)
        for fid, sid, fname, sname, site, item, lon_site, lat_site in records:
            f.write(f"{fid}\t{sid}\t{fname}\t{sname}\t{site}\t{item}\t{lon_site:.2f}\t{lat_site:.2f}\n")

    print(f"输出完成：{out_path}")
    print(f"异常场地输入行数：{len(abn_df)}")
    print(f"输出记录行数（5 km 内最近断层段）：{len(records)}")


if __name__ == "__main__":
    main()

