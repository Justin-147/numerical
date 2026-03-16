# GNSS 基线长度与方位角分析（GNSS-baseline）

本说明文档专门针对本目录中的 `GNSS-baseline.py` 程序，实现**GNSS基线长度与方位角**的计算与成图流程，用于研究构造运动相关异常。

## 1. 主程序

- **主程序**：`GNSS-baseline.py`
- **依赖安装**：
  ```bash
  pip install -r GNSS-baseline_requirements.txt
  ```

## 2. 数据说明

- 源数据格式与 GNSS-Data 目录中的 CENC GNSS 文件一致（如 GNSS-Data`/SCLH_raw.neu`、`SCTQ_raw.neu`）：
  - 第 1 行：`#Reference position  lon  lat  height  STATION`
  - 第 2 行：`# YYYYMMDD YYYY.DECM N(mm) E(mm) U(mm) sig_n(mm) sig_e(mm) sig_u(mm)`
  - 其后为逐行观测数据。
- 默认从GNSS-Data`/` 下读取原始 `.neu` 文件。

## 3. 功能概览

对给定的两个 GNSS 台站（如 `SCLH_raw.neu` 与 `SCTQ_raw.neu`）：

1. 读取两站 CENC 格式时序，解析参考经纬度与台站名；
2. 按十进制年（`YYYY.DECM`）对齐，仅保留共同时间段（对齐前对 DECM 四舍五入到 4 位小数，避免不同文件浮点表示差异导致匹配不全）；
3. 使用 WGS‑84 椭球算法，逐历元计算：
  - 大地线长度 `S`（m）
  - 正向方位角 `A1`（0–360°）；
4. 将长度与方位角转换为相对变化量并统一单位：
  - baseline：减去首历元长度，转为 **mm**；
  - azimuth：减去首历元方位角，转为 **millisecond of degree**；
5. 对 baseline 与 azimuth 分别做线性拟合并扣除趋势，得到去趋势序列；
6. 输出文本与图件，便于后续破裂前异常分析或与其它测项联合对比。

## 4. 输出结果

对每一对台站 `<STA1>_<STA2>`，`GNSS-baseline.py` 会在本目录下 GNSS-Out`/` 中生成：

- `STA1_STA2_baseline.txt`：
  - 每行 `YYYY.DECM  baseline(mm)  azimuth(millisecond)`，为扣除初始值后的相对变化量；
- `STA1_STA2_baseline_detrend.txt`：
  - 在上述基础上进一步扣除线性趋势；
- 图件（仅 PNG）：
  - `*_baseline.png`：baseline/mm 原始时间序列；
  - `*_azimuth.png`：azimuth/millisecond 原始时间序列；
  - `*_baseline_detrend.png`：baseline/mm 去趋势；
  - `*_azimuth_detrend.png`：azimuth/millisecond 去趋势。

## 5. 使用方式

在 `GNSS-baseline.py` 末尾的 `main()` 中配置站对列表：

```python
station_pairs = [
    ("SCLH_raw.neu", "SCTQ_raw.neu"),
    ("YNYL_raw.neu", "YNYS_raw.neu"),
    ("GSDX_raw.neu", "NXHY_raw.neu"),
]
```

然后在 `numerical/fault-movement-anomaly` 目录下运行：

```bash
cd fault-movement-anomaly
python GNSS-baseline.py
```

程序会自动从 GNSS-Data`/` 读取同名文件，并将结果写入本目录的 GNSS-Out`/` 文件夹中。