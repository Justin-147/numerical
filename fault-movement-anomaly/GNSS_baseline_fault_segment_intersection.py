#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GNSS_baseline_fault_segment_intersection.py
==========================================

读取：
1) FaultCord_justExample.xlsx
   - 示例断层段迹线点位（每条断层段是一条折线：多点坐标）
2) GNSS基线异常表-示例.xlsx
   - 示例异常 GNSS 基线站点对列表

判断：每个异常 GNSS 基线（两站连线）与哪些断层段折线相交。

输出：Abnormal_Fault_Segments_from_GNSS_Baseline.txt
字段：断层编号、断层段编号、断层名称、断层段名称、GNSS基线异常站点对
若同一断层段与多个基线相交，分多行输出。
若没有任何相交，也会生成输出文件（仅表头）。

依赖：pandas、openpyxl、numpy（用于读 xlsx）
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StationCoord:
    name: str
    lon: float
    lat: float

# 输入/输出文件名
FAULT_XLSX = "FaultCord_justExample.xlsx" # 断层段迹线点位（每条断层段是一条折线：多点坐标）
ABNORMAL_XLSX = "GNSS基线异常表-示例.xlsx" # 异常 GNSS 基线站点对列表
OUTPUT_TXT = "Abnormal_Fault_Segments_from_GNSS_Baseline.txt" # 异常断层段—异常基线对”的对应关系表

# 固定表头（必须存在）
FAULT_COLUMNS = {
    "fault_id": "断层编号",
    "segment_id": "断层段编号",
    "fault_name": "断层名称",
    "segment_name": "断层段名称",
    "lon": "经度",
    "lat": "纬度",
}
ABNORMAL_COLUMNS = {
    "sta1_name": "站点1名称",
    "sta1_lon": "站点1经度",
    "sta1_lat": "站点1纬度",
    "sta2_name": "站点2名称",
    "sta2_lon": "站点2经度",
    "sta2_lat": "站点2纬度",
}

OUTPUT_HEADER = "#fault_id\tsegment_id\tfault_name\tsegment_name\tabnormal_gnss_baseline_pair\n"


def _norm_col(s: str) -> str:
    return re.sub(r"\s+", "", str(s).strip()).lower()


def _pick_col(columns: Iterable[str], candidates: List[str]) -> Optional[str]:
    cols = list(columns)
    norm = {_norm_col(c): c for c in cols}
    for cand in candidates:
        if _norm_col(cand) in norm:
            return norm[_norm_col(cand)]
    return None


def _read_fault_segments(xlsx: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx, sheet_name=0)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _read_abnormal_pairs(xlsx: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx, sheet_name=0)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _get_station_coord_from_row(
    row: pd.Series,
    sta_col: str,
    lon_col: str,
    lat_col: str,
) -> StationCoord:
    name = str(row[sta_col]).strip()
    if pd.isna(row[lon_col]) or pd.isna(row[lat_col]):
        raise ValueError(f"异常表缺少站点经纬度：{name}（请补齐 {lon_col}/{lat_col}）")
    return StationCoord(name=name, lon=float(row[lon_col]), lat=float(row[lat_col]))


def _orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _on_segment(a: np.ndarray, b: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> bool:
    if abs(_orient(a, b, p)) > eps:
        return False
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def segments_intersect(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float], d: Tuple[float, float]) -> bool:
    """
    线段 AB 与 CD 是否相交（含端点接触与共线重叠）。
    坐标在 lon/lat 平面中近似处理。
    """
    A = np.array(a, dtype=float)
    B = np.array(b, dtype=float)
    C = np.array(c, dtype=float)
    D = np.array(d, dtype=float)

    o1 = _orient(A, B, C)
    o2 = _orient(A, B, D)
    o3 = _orient(C, D, A)
    o4 = _orient(C, D, B)

    eps = 1e-12

    # 一般相交
    if (o1 > eps and o2 < -eps or o1 < -eps and o2 > eps) and (o3 > eps and o4 < -eps or o3 < -eps and o4 > eps):
        return True

    # 处理共线/端点
    if abs(o1) <= eps and _on_segment(A, B, C, eps):
        return True
    if abs(o2) <= eps and _on_segment(A, B, D, eps):
        return True
    if abs(o3) <= eps and _on_segment(C, D, A, eps):
        return True
    if abs(o4) <= eps and _on_segment(C, D, B, eps):
        return True

    return False


def polyline_intersects_segment(poly: np.ndarray, seg_a: Tuple[float, float], seg_b: Tuple[float, float]) -> bool:
    """折线 poly（n×2）是否与线段 AB 相交。"""
    if poly.shape[0] < 2:
        return False
    for i in range(poly.shape[0] - 1):
        c = (float(poly[i, 0]), float(poly[i, 1]))
        d = (float(poly[i + 1, 0]), float(poly[i + 1, 1]))
        if segments_intersect(seg_a, seg_b, c, d):
            return True
    return False


def main() -> None:
    work_dir = Path(__file__).resolve().parent
    fault_xlsx = (work_dir / FAULT_XLSX).resolve()
    abnormal_xlsx = (work_dir / ABNORMAL_XLSX).resolve()
    out_path = (work_dir / OUTPUT_TXT).resolve()

    fault_df = _read_fault_segments(fault_xlsx)
    abn_df = _read_abnormal_pairs(abnormal_xlsx)

    # 固定表头检查
    for k, col in FAULT_COLUMNS.items():
        if col not in fault_df.columns:
            raise ValueError(f"断层 Excel 缺少固定列：{col}（key={k}）")
    for k, col in ABNORMAL_COLUMNS.items():
        if col not in abn_df.columns:
            raise ValueError(f"GNSS 异常表缺少固定列：{col}（key={k}）")

    fault_id_col = FAULT_COLUMNS["fault_id"]
    seg_id_col = FAULT_COLUMNS["segment_id"]
    fault_name_col = FAULT_COLUMNS["fault_name"]
    seg_name_col = FAULT_COLUMNS["segment_name"]
    lon_col = FAULT_COLUMNS["lon"]
    lat_col = FAULT_COLUMNS["lat"]

    sta1_col = ABNORMAL_COLUMNS["sta1_name"]
    sta2_col = ABNORMAL_COLUMNS["sta2_name"]
    sta1_lon_col = ABNORMAL_COLUMNS["sta1_lon"]
    sta1_lat_col = ABNORMAL_COLUMNS["sta1_lat"]
    sta2_lon_col = ABNORMAL_COLUMNS["sta2_lon"]
    sta2_lat_col = ABNORMAL_COLUMNS["sta2_lat"]

    # ---- 预处理断层段折线 ----
    seg_groups = fault_df.groupby([fault_id_col, seg_id_col], dropna=False)
    fault_segments: List[Tuple[str, str, str, str, np.ndarray]] = []
    for (fid, sid), g in seg_groups:
        g2 = g.dropna(subset=[lon_col, lat_col])
        if g2.shape[0] < 2:
            continue
        coords = g2[[lon_col, lat_col]].to_numpy(dtype=float)
        fname = str(g2[fault_name_col].iloc[0]) if fault_name_col in g2.columns else ""
        sname = str(g2[seg_name_col].iloc[0]) if seg_name_col in g2.columns else ""
        fault_segments.append((str(fid), str(sid), fname, sname, coords))

    # ---- 逐基线检查相交 ----
    records: List[Tuple[str, str, str, str, str]] = []
    for _, row in abn_df.iterrows():
        sta1 = _get_station_coord_from_row(row, sta1_col, sta1_lon_col, sta1_lat_col)
        sta2 = _get_station_coord_from_row(row, sta2_col, sta2_lon_col, sta2_lat_col)
        seg_a = (sta1.lon, sta1.lat)
        seg_b = (sta2.lon, sta2.lat)
        pair_name = f"{sta1.name}-{sta2.name}"

        for fid, sid, fname, sname, poly in fault_segments:
            if polyline_intersects_segment(poly, seg_a, seg_b):
                records.append((fid, sid, fname, sname, pair_name))

    # ---- 输出 ----
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(OUTPUT_HEADER)
        for fid, sid, fname, sname, pair in records:
            f.write(f"{fid}\t{sid}\t{fname}\t{sname}\t{pair}\n")

    uniq_segments = {(fid, sid) for fid, sid, _, _, _ in records}
    print(f"异常断层段数量（去重）：{len(uniq_segments)}")
    for fid, sid, fname, sname, pair in records:
        print(f"{fid}\t{sid}\t{fname}\t{sname}\t{pair}")
    print(f"\n输出完成：{out_path}")
    print(f"相交记录数（基线-断层段配对行数）：{len(records)}")


if __name__ == "__main__":
    main()

