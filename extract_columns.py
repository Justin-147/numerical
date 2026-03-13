# -*- coding: utf-8 -*-
"""
多列数据文件按列提取并输出。
支持：有/无表头、中文或英文表头、指定列号（从1开始）抽取、是否输出表头。
"""

from pathlib import Path

# ========= 可修改参数 =========
""" INPUT_FILE = Path(r"d:\numerical\cycle-related-anomaly\32016_3_2221_processed.txt")  # 输入文件路径
OUTPUT_FILE = Path(r"d:\numerical\R-value\32016_3_2221_col1_4.txt")                          # 输出文件路径
COLUMNS = [1, 4]           # 要抽取的列号，从 1 开始，如 [1, 4] 表示第 1 列和第 4 列 """
INPUT_FILE = Path(r"d:\numerical\trend-related-anomaly\14001_1_2231_processed.txt")  # 输入文件路径
OUTPUT_FILE = Path(r"d:\numerical\R-value\14001_1_2231_trend_related_anomaly.txt")                          # 输出文件路径
COLUMNS = [1, 2]           # 要抽取的列号，从 1 开始，如 [1, 4] 表示第 1 列和第 4 列

HAS_HEADER = True          # 输入文件是否有表头行
OUTPUT_HEADER = False      # 输出文件是否写入表头（仅当 HAS_HEADER 为 True 时有效）
ENCODING = "utf-8"         # 输入文件编码，中文表头可能是 gbk，可改为 "gbk"
DELIMITER = "\t"           # 列分隔符，"\t" 为制表符，" " 为空格，None 表示按任意空白切分
# =============================


def _split_line(line: str, delim):
    if delim is None:
        return line.split()
    return line.split(delim)


def main():
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)
    if not input_path.exists():
        print(f"错误：输入文件不存在 {input_path}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 尝试解码
    try:
        with open(input_path, encoding=ENCODING) as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(input_path, encoding="gbk") as f:
            lines = f.readlines()

    if not lines:
        print("错误：输入文件为空")
        return

    # 解析：第一行可能是表头
    if HAS_HEADER and len(lines) > 0:
        header_line = lines[0].rstrip("\n\r")
        data_lines = lines[1:]
        header_cols = _split_line(header_line, DELIMITER)
    else:
        header_cols = []
        data_lines = lines

    ncols_in = None
    out_rows = []

    for i, line in enumerate(data_lines):
        line = line.rstrip("\n\r")
        if not line.strip():
            continue
        cols = _split_line(line, DELIMITER)
        if ncols_in is None:
            ncols_in = len(cols)
        max_col = max(COLUMNS)
        if max_col > len(cols):
            print(f"警告：第 {i+1} 行只有 {len(cols)} 列，请求列号最大为 {max_col}，已跳过该行")
            continue
        selected = [cols[c - 1] for c in COLUMNS]
        out_rows.append(selected)

    if HAS_HEADER and OUTPUT_HEADER and header_cols:
        max_col = max(COLUMNS)
        if max_col <= len(header_cols):
            header_selected = [header_cols[c - 1] for c in COLUMNS]
            header_str = "\t".join(header_selected)
        else:
            header_str = "\t".join(f"Col{c}" for c in COLUMNS)
    else:
        header_str = None

    with open(output_path, "w", encoding="utf-8") as f:
        if header_str is not None:
            f.write(header_str + "\n")
        for row in out_rows:
            f.write("\t".join(row) + "\n")

    print(f"已从 {input_path.name} 抽取列 {COLUMNS}，共 {len(out_rows)} 行，写入 {output_path}")


if __name__ == "__main__":
    main()
