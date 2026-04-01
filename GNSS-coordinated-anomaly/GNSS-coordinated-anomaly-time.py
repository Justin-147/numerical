#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNSS 协调方向异常：时间相关（站点对滑动相关系数）

输入（来自 GNSS-coordinated-anomaly-filt.py 的输出）：
- FiltDataOut/*_HHTfilt.txt

功能：
- 给定站点对（例如 LS06 LS07），在给定窗长/步长下，计算站点对间 N/E/U 的滑动 Pearson 相关系数序列
- 输出到 TimeCorrelationOut/，文件名形如：LS06-LS07-TimeCorrelation.txt
- 时间戳取“窗尾”日期（YYYYMMDD）

直接运行：
- 修改下面「默认配置」区块中的站点对、窗长、路径后，双击或 `python GNSS-coordinated-anomaly-time.py` 即可；
- 需要时用命令行参数覆盖（如 `--pair A B`、`--no-plot` 关闭绘图）。
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np


# ===========================================================================
# 默认配置（直接运行脚本时使用；改这里即可）
# ===========================================================================
DEFAULT_IN_PATH = os.path.join("GNSS-coordinated-anomaly", "FiltDataOut") # 滤波输出目录
DEFAULT_OUT_DIR = os.path.join("GNSS-coordinated-anomaly", "TimeCorrelationOut") # 输出目录
DEFAULT_WINDOW_DAYS = 30 # 窗长（天）
DEFAULT_STEP_DAYS = 1 # 滑动步长（天）

# 是否在写出 txt 后绘制 N/E/U 相关系数时间序列（上中下三子图），输出同名 .png
DEFAULT_ENABLE_PLOT = True # 是否绘图

# 站点对列表：每项为 (站点1, 站点2)，输出文件名 <站1>-<站2>-TimeCorrelation.txt
DEFAULT_SITE_PAIRS: List[Tuple[str, str]] = [
    ("LS06", "LS07"),
    ("LS05", "LS06"),
]
# ===========================================================================


def _load_hhtfilt(site: str, in_path: str) -> Dict[str, np.ndarray]:
    fp = os.path.join(in_path, f"{site}_HHTfilt.txt")
    if not os.path.isfile(fp):
        raise FileNotFoundError(f"未找到站点文件：{fp}")
    data = np.loadtxt(fp, delimiter="\t", comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 5:
        raise ValueError(f"列数不足（期望>=5：YYYYMMDD, YYYY.DECM, N,E,U...）：{fp}")
    days = data[:, 0].astype(np.int32)
    n_mm = data[:, 2].astype(np.float64)
    e_mm = data[:, 3].astype(np.float64)
    u_mm = data[:, 4].astype(np.float64)
    # 确保按日期升序
    if days.size >= 2 and np.any(np.diff(days) < 0):
        idx = np.argsort(days)
        days = days[idx]
        n_mm = n_mm[idx]
        e_mm = e_mm[idx]
        u_mm = u_mm[idx]
    return {"days": days, "N": n_mm, "E": e_mm, "U": u_mm}


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    # 常量序列会导致相关系数为 NaN（与 numpy.corrcoef 一致）
    if a.size < 2:
        return float("nan")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float("nan")
    if float(np.nanstd(a)) == 0.0 or float(np.nanstd(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _align_by_days(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    days_a = a["days"]
    days_b = b["days"]
    days, ia, ib = np.intersect1d(days_a, days_b, assume_unique=False, return_indices=True)
    aa = {"N": a["N"][ia], "E": a["E"][ia], "U": a["U"][ia]}
    bb = {"N": b["N"][ib], "E": b["E"][ib], "U": b["U"][ib]}
    return days.astype(np.int32), aa, bb


def _plot_correlation_series(
    out_png: str,
    site1: str,
    site2: str,
    ymds: np.ndarray,
    n_corr: np.ndarray,
    e_corr: np.ndarray,
    u_corr: np.ndarray,
    window_days: int,
) -> None:
    from datetime import datetime

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    x = [datetime.strptime(str(int(d)), "%Y%m%d") for d in ymds]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["N correlation", "E correlation", "U correlation"]
    for ax, y, lab in zip(axes, (n_corr, e_corr, u_corr), labels):
        ax.plot(x, y, lw=0.9, color="C0")
        ax.axhline(0.0, color="gray", lw=0.6, ls="--")
        ax.set_ylabel(lab)
        ax.set_ylim(-1.05, 1.05)
    axes[0].set_title(f"{site1}-{site2} sliding Pearson r (window={window_days} d, end date)")
    axes[-1].set_xlabel("Date (window end)")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def run_one_pair(
    site1: str,
    site2: str,
    *,
    in_path: str,
    out_dir: str,
    window_days: int,
    step_days: int,
    enable_plot: bool,
) -> Tuple[str, str | None]:
    a = _load_hhtfilt(site1, in_path)
    b = _load_hhtfilt(site2, in_path)
    days, aa, bb = _align_by_days(a, b)

    if window_days <= 1:
        raise ValueError("window_days 必须 >= 2")
    if step_days <= 0:
        raise ValueError("step_days 必须 >= 1")
    if days.size < window_days:
        raise ValueError(f"{site1}-{site2} 共同日期数不足：{days.size} < window_days({window_days})")

    os.makedirs(out_dir, exist_ok=True)
    out_fp = os.path.join(out_dir, f"{site1}-{site2}-TimeCorrelation.txt")

    ymds: List[int] = []
    n_list: List[float] = []
    e_list: List[float] = []
    u_list: List[float] = []
    for s in range(0, days.size - window_days + 1, step_days):
        e = s + window_days
        ymd = int(days[e - 1])  # 窗尾日期
        n_c = _pearson_corr(aa["N"][s:e], bb["N"][s:e])
        e_c = _pearson_corr(aa["E"][s:e], bb["E"][s:e])
        u_c = _pearson_corr(aa["U"][s:e], bb["U"][s:e])
        ymds.append(ymd)
        n_list.append(n_c)
        e_list.append(e_c)
        u_list.append(u_c)

    with open(out_fp, "w", encoding="utf-8", newline="\n") as f:
        f.write("# YYYYMMDD\tN_correlation\tE_correlation\tU_correlation\n")
        for ymd, n_c, e_c, u_c in zip(ymds, n_list, e_list, u_list):
            f.write(f"{ymd}\t{n_c:.10g}\t{e_c:.10g}\t{u_c:.10g}\n")

    out_png: str | None = None
    if enable_plot and ymds:
        out_png = os.path.join(out_dir, f"{site1}-{site2}-TimeCorrelation.png")
        _plot_correlation_series(
            out_png,
            site1,
            site2,
            np.asarray(ymds, dtype=np.int64),
            np.asarray(n_list, dtype=np.float64),
            np.asarray(e_list, dtype=np.float64),
            np.asarray(u_list, dtype=np.float64),
            window_days,
        )

    return out_fp, out_png


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="GNSS-coordinated-anomaly-time.py",
        description="站点对滑动相关系数（N/E/U）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--in-path", default=DEFAULT_IN_PATH, help="滤波输出目录（包含 *_HHTfilt.txt）")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="输出目录")
    p.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS, help="窗长（天）")
    p.add_argument("--step-days", type=int, default=DEFAULT_STEP_DAYS, help="滑动步长（天）")
    p.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("SITE1", "SITE2"),
        help="站点对（可重复多次）；不传则使用脚本顶部的 DEFAULT_SITE_PAIRS",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="关闭绘图（默认按脚本顶部 DEFAULT_ENABLE_PLOT，且默认开启）",
    )
    return p


def main(argv: List[str] | None = None) -> int:
    import sys

    if argv is None:
        argv = sys.argv[1:]
    args = build_parser().parse_args(argv)
    pairs: List[Tuple[str, str]] = list(args.pair) if args.pair else list(DEFAULT_SITE_PAIRS)
    if not pairs:
        raise SystemExit("未配置站点对：请在脚本顶部设置 DEFAULT_SITE_PAIRS，或使用 --pair SITE1 SITE2")

    enable_plot = bool(DEFAULT_ENABLE_PLOT) and not bool(args.no_plot)

    for s1, s2 in pairs:
        out_fp, out_png = run_one_pair(
            s1,
            s2,
            in_path=str(args.in_path),
            out_dir=str(args.out_dir),
            window_days=int(args.window_days),
            step_days=int(args.step_days),
            enable_plot=enable_plot,
        )
        msg = f"{s1}-{s2}: {out_fp}"
        if out_png:
            msg += f" | {out_png}"
        print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

