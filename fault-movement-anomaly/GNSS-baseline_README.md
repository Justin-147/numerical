# GNSS 基线长度与方位角分析（GNSS-baseline）

本说明文档专门针对本目录中的 `GNSS-baseline.py` 程序，实现**GNSS基线长度与方位角**的计算与成图流程，用于研究构造运动相关异常。

## 1. 主程序

- **主程序**：`GNSS-baseline.py`
- **依赖安装**：
  ```bash
  pip install -r GNSS-baseline_requirements.txt
  ```

## 2. 数据说明

- 源数据格式与 `GNSS-Data/` 目录中的 CENC GNSS 文件一致（如 `GNSS-Data/YNYA_raw.neu`、`GNSS-Data/XIAG_raw.neu`）：
  - 第 1 行：`#Reference position  lon  lat  height  STATION`
  - 第 2 行：`# YYYYMMDD YYYY.DECM N(mm) E(mm) U(mm) sig_n(mm) sig_e(mm) sig_u(mm)`
  - 其后为逐行观测数据。
- 默认从 `GNSS-Data/` 下读取原始 `.neu` 文件。

## 3. 功能概览

对给定的两个 GNSS 台站（如 `YNYA_raw.neu` 与 `XIAG_raw.neu`）：

1. 读取两站 CENC 格式时序，解析参考经纬度与台站名；
2. 按十进制年（`YYYY.DECM`）对齐，仅保留共同时间段（对齐前对 DECM 四舍五入到 4 位小数，避免不同文件浮点表示差异导致匹配不全）；
3. 使用 WGS‑84 椭球算法，逐历元计算：
  - 大地线长度 `S`（m）
  - 正向方位角 `A1`（0–360°）；
4. 将长度与方位角转换为相对变化量并统一单位：
  - baseline：减去首历元长度，转为 **mm**；
  - azimuth：减去首历元方位角，转为 **millisecond of degree**；
5. 对 baseline 与 azimuth 分别做线性拟合并扣除趋势，得到去趋势序列；
6. **一阶误差传播**：使用输入文件中的 `sig_n`、`sig_e`（mm），假定两站 N/E 误差相互独立。按与主流程一致的变换 `N/E → 近似经纬度 → 大地线 S 与方位角 A1` 做链式求导（数值雅可比），得到每历元的 **绝对不确定度** σ_S（m）、σ_A1（度），并换算输出为 σ_baseline(mm)、σ_azimuth(millisecond)。
7. 输出文本与图件；当前传播未计入参考坐标、U 分量、相关性及垂向影响，需更严格不确定度时请自行扩展。

## 4. 输出结果

对每一对台站 `<STA1>_<STA2>`，`GNSS-baseline.py` 会在本目录下 `GNSS-Out/` 中生成：

- `STA1_STA2_baseline.txt`：
  - 每行：`YYYY.DECM`、`baseline(mm)`、`azimuth(millisecond)`、`sigma_baseline(mm)`、`sigma_azimuth(millisecond)`；
- `STA1_STA2_baseline_detrend.txt`：
  - 在上述基础上进一步扣除线性趋势；**σ 列与去趋势前列相同**（近似，未单独推导线拟合残差的不确定度）；
- 图件（仅 PNG）：
  - `*_baseline.png`：baseline/mm 原始时间序列；
  - `*_azimuth.png`：azimuth/millisecond 原始时间序列；
  - `*_baseline_detrend.png`：baseline/mm 去趋势；
  - `*_azimuth_detrend.png`：azimuth/millisecond 去趋势。
  - 在 `*_azimuth.png` 左上角标注 `Start Azimuth = xxxx.xxxx°`（首历元 A1，保留 4 位小数）。

## 5. 使用方式

在 `GNSS-baseline.py` 末尾的 `main()` 中配置站对列表：

```python
station_pairs = [
    ("YNYA_raw.neu", "XIAG_raw.neu"),
    ("YNGM_raw.neu", "YNRL_raw.neu"),
    ("SCTQ_raw.neu", "SCMB_raw.neu"),
    ("YNTH_raw.neu", "YNJP_raw.neu"),
]
```

然后在 `numerical/fault-movement-anomaly` 目录下运行：

```bash
cd fault-movement-anomaly
python GNSS-baseline.py
```

程序会自动从 `GNSS-Data/` 读取同名文件，并将结果写入本目录的 `GNSS-Out/` 文件夹中。