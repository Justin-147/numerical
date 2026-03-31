#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CrossFault-baseline.py
======================

读取跨断层基线或水准观测的两列数据（时间 + 数值），按给定“月窗长”逐点计算差分值：
  diff(tA) = value(tA) - value(tB)

其中 B 为 A 之前、且与 A 的“年月差”为窗长（月）的观测点（不严格卡天）。
若目标年月没有任何观测，则该点不输出。
若目标年月同月有多天观测，则选择“与窗长更接近”的那个点：
  令 tB' = tB + 窗长（月）后的日期（同日，必要时按月末截断）
  选择使 |tA - tB'| 最小的 tB

输入文件可能无表头，分隔符可为逗号、空格或 tab。
时间格式固定为 8 位 yyyymmdd。

默认：
  - 输入目录：CrossFault-Data/
  - 输出目录：CrossFault-Out/
  - 处理目录下全部 .txt
  - 窗长：12 个月
"""

from __future__ import annotations

import calendar
import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator


# ============
# 可修改参数区
# ============

DATA_DIR = Path(__file__).resolve().parent / "CrossFault-Data" # 输入目录
OUT_DIR = Path(__file__).resolve().parent / "CrossFault-Out" # 输出目录

# 若只想处理某一个文件，把 FILE_GLOB 改成具体文件名（例如 "下关水2-水1.txt"）
FILE_GLOB = "*.txt"

# 差分窗长（月）：可改为 3/6/12/24 等，可选
WINDOW_MONTHS = 12

# 字体：Times New Roman（若系统缺失会自动回退）
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False


@dataclass(frozen=True)
class Obs:
    t: date
    y: float
    tcode: str  # 原始 yyyymmdd 字符串


_SPLIT_RE = re.compile(r"[,\t ]+")
_DATE_RE = re.compile(r"^\d{8}$")


def _parse_yyyymmdd(s: str) -> date:
    s = s.strip()
    if not _DATE_RE.match(s):
        raise ValueError(f"时间格式不是 8 位 yyyymmdd: {s!r}")
    y = int(s[0:4])
    m = int(s[4:6])
    d = int(s[6:8])
    return date(y, m, d)


def _add_months(dt: date, months: int) -> date:
    """dt 加 months 个月；若目标月天数不足，则按月末截断。"""
    y = dt.year
    m = dt.month + months
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    last_day = calendar.monthrange(y, m)[1]
    d = min(dt.day, last_day)
    return date(y, m, d)


def _read_two_col_file(path: Path) -> List[Obs]:
    """
    读取两列：yyyymmdd, value
    支持逗号/空格/tab 分隔；可有表头（无法解析为 8 位日期的行会跳过）。
    """
    obs: List[Obs] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p for p in _SPLIT_RE.split(line) if p]
            if len(parts) < 2:
                continue
            tcode = parts[0].strip()
            if not _DATE_RE.match(tcode):
                # 可能是表头，跳过
                continue
            try:
                t = _parse_yyyymmdd(tcode)
                y = float(parts[1])
            except Exception:
                continue
            obs.append(Obs(t=t, y=y, tcode=tcode))

    obs.sort(key=lambda o: o.t)
    return obs


def _compute_month_diff(obs: List[Obs], window_months: int) -> List[Tuple[str, float]]:
    """
    返回列表：[(tA_code, diff), ...]
    """
    if window_months <= 0:
        raise ValueError("WINDOW_MONTHS 必须为正整数（月）")

    # year-month -> indices（按时间排序后）
    month_map: Dict[Tuple[int, int], List[int]] = {}
    for i, o in enumerate(obs):
        key = (o.t.year, o.t.month)
        month_map.setdefault(key, []).append(i)

    out: List[Tuple[str, float]] = []
    for i, a in enumerate(obs):
        target = _add_months(a.t, -window_months)
        key = (target.year, target.month)
        cand_idx = month_map.get(key)
        if not cand_idx:
            continue

        best_j = None
        best_score = None
        for j in cand_idx:
            if j >= i:
                break  # month_map 内 i 递增，已到未来
            b = obs[j]
            b_plus = _add_months(b.t, window_months)
            score = abs((a.t - b_plus).days)
            if best_score is None or score < best_score:
                best_score = score
                best_j = j

        if best_j is None:
            continue

        b = obs[best_j]
        out.append((a.tcode, a.y - b.y))

    return out


def main() -> None:
    in_dir = DATA_DIR
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(FILE_GLOB))
    if not files:
        raise FileNotFoundError(f"未找到输入文件：{in_dir / FILE_GLOB}")

    for path in files:
        obs = _read_two_col_file(path)
        if len(obs) < 2:
            print(f"跳过（有效数据行不足）：{path.name}")
            continue

        diffs = _compute_month_diff(obs, WINDOW_MONTHS)
        stem = path.stem
        out_name = f"{stem}_{WINDOW_MONTHS}月窗长差分.txt"
        out_path = out_dir / out_name
        title = f"{stem}_{WINDOW_MONTHS}月窗长差分"

        with out_path.open("w", encoding="utf-8") as f:
            for tcode, dv in diffs:
                f.write(f"{tcode}\t{dv:.10g}\n")

        # 绘图与存图（PNG）
        if diffs:
            x = [datetime.strptime(t, "%Y%m%d") for t, _ in diffs]
            y = [float(v) for _, v in diffs]
            y_min = min(y)
            y_max = max(y)
            pad = (y_max - y_min) / 10.0 if y_max != y_min else (abs(y_max) / 10.0 if y_max != 0 else 1.0)

            fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
            # 青色空心小圆圈 + 红色细线连接
            ax.plot(x, y, color="r", linewidth=0.8, zorder=1)
            ax.plot(
                x,
                y,
                linestyle="None",
                marker="o",
                markersize=3.0,
                markerfacecolor="none",
                markeredgecolor="c",
                markeredgewidth=0.8,
                zorder=2,
            )
            ax.set_xlabel("Date", fontfamily="Times New Roman")
            ax.set_ylabel("mm", fontfamily="Times New Roman")
            # title 允许中英文混排：优先 Times New Roman，不支持时回退到中文字体
            ax.set_title(
                title,
                fontfamily=["Times New Roman", "Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS", "DejaVu Sans"],
            )
            ax.set_ylim(y_min - pad, y_max + pad)

            # 主刻度：每年 1 月 1 日；根据 x 轴范围自动稀疏，主 tick+label 不超过 10 个
            span_years = max(x).year - min(x).year + 1
            year_step = max(1, int(math.ceil(span_years / 10)))
            ax.xaxis.set_major_locator(mdates.YearLocator(base=year_step, month=1, day=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            # 次刻度：同样按年标注（每年 1 月 1 日），但不显示 ticklabel
            ax.xaxis.set_minor_locator(mdates.YearLocator(base=1, month=1, day=1))

            ax.tick_params(axis="x", which="major", length=6)
            ax.tick_params(axis="x", which="minor", length=3, labelbottom=False)

            # y 轴刻度：不超过 10 个，避免 label 重叠
            ax.yaxis.set_major_locator(MaxNLocator(nbins=10))

            # tick 字体：Times New Roman
            for lab in ax.get_xticklabels(which="both") + ax.get_yticklabels(which="both"):
                lab.set_fontfamily("Times New Roman")

            fig.tight_layout()
            png_path = out_dir / f"{title}.png"
            fig.savefig(png_path, dpi=300)
            plt.close(fig)
            print(f"图件输出：{png_path.name}")

        print(f"完成：{path.name} -> {out_path.name}（输出 {len(diffs)} 行）")


if __name__ == "__main__":
    main()

