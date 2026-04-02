#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量按日期范围截取 DataIn 中的 *_raw.neu 文件。

行为：
- 对每个匹配文件：
  - 先把原文件重命名为 <原文件名>.allbak（若已存在则报错退出，避免覆盖）
  - 再用原文件名写出“截取后的新文件”（保留原表头行）

默认日期范围：20120401 ~ 20130430（含）

用法：
  python GNSS-coordinated-anomaly/cut_datain_by_date.py
  python GNSS-coordinated-anomaly/cut_datain_by_date.py --start 20110101 --end 20161231
  python GNSS-coordinated-anomaly/cut_datain_by_date.py --data-dir GNSS-coordinated-anomaly/DataIn --glob "*_raw.neu"
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List, Tuple


def _is_yyyymmdd(s: str) -> bool:
    if len(s) != 8 or not s.isdigit():
        return False
    y = int(s[0:4])
    m = int(s[4:6])
    d = int(s[6:8])
    if not (1900 <= y <= 2100):
        return False
    if not (1 <= m <= 12):
        return False
    if not (1 <= d <= 31):
        return False
    return True


def _cut_one_file(path: str, start: int, end: int) -> Tuple[int, int]:
    """
    返回 (kept_lines, total_data_lines)。
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    header: List[str] = []
    data: List[str] = []
    for ln in lines:
        if ln.startswith("#"):
            header.append(ln)
        else:
            if ln.strip():
                data.append(ln)

    kept: List[str] = []
    total = 0
    for ln in data:
        total += 1
        parts = ln.split()
        if not parts:
            continue
        ymd = parts[0]
        if not _is_yyyymmdd(ymd):
            # 非数据行就原样丢弃（更安全）
            continue
        v = int(ymd)
        if start <= v <= end:
            kept.append(ln)

    bak = path + ".allbak"
    if os.path.exists(bak):
        raise FileExistsError(f"备份文件已存在，拒绝覆盖：{bak}")

    os.replace(path, bak)

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for ln in header:
            f.write(ln.rstrip("\n") + "\n")
        for ln in kept:
            f.write(ln.rstrip("\n") + "\n")

    return len(kept), total


def main() -> None:
    p = argparse.ArgumentParser(description="按日期范围截取 DataIn *_raw.neu，并备份为 .allbak")
    p.add_argument("--data-dir", default=os.path.join("GNSS-coordinated-anomaly", "DataIn"), help="输入目录")
    p.add_argument("--glob", dest="glob_pattern", default="*_raw.neu", help="文件匹配模式（默认 *_raw.neu）")
    p.add_argument("--start", type=int, default=20120401, help="起始日期 YYYYMMDD（含）")
    p.add_argument("--end", type=int, default=20130430, help="结束日期 YYYYMMDD（含）")
    args = p.parse_args()

    if args.start > args.end:
        raise ValueError("start 不能大于 end")
    if not _is_yyyymmdd(str(args.start)) or not _is_yyyymmdd(str(args.end)):
        raise ValueError("start/end 必须是 YYYYMMDD")

    paths = sorted(glob.glob(os.path.join(args.data_dir, args.glob_pattern)))
    if not paths:
        print(f"未找到匹配文件：{args.data_dir} / {args.glob_pattern}")
        return

    print(f"范围：[{args.start}, {args.end}]，文件数：{len(paths)}")
    for i, path in enumerate(paths, 1):
        name = os.path.basename(path)
        kept, total = _cut_one_file(path, args.start, args.end)
        print(f"[{i}/{len(paths)}] {name}: kept {kept}/{total} -> {name}（原文件备份为 {name}.allbak）")


if __name__ == "__main__":
    main()

