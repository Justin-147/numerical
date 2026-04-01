# GNSS-coordinated-anomaly（HHT 频带滤波 + 空间格网 + 站点对时间相关）

## 目标

把 CENC 样式 `*_raw.neu`（逐日 NEU 位移）转换为 **20–150 天游程频带**的 N/E/U “滤波”结果（基于 EEMD + Hilbert 瞬时频率筛选），并可选做空间格网统计与出图，以及站点对滑动相关系数分析。

## 输入数据格式

程序默认读取 `GNSS-coordinated-anomaly/DataIn/*.neu`（建议命名 `*_raw.neu`），文件头两行形如：

```text
#Reference position    103.39     30.11    0.00    LS01
# YYYYMMDD YYYY.DECM     N(mm)     E(mm)     U(mm) sig_n(mm) sig_e(mm) sig_u(mm)
20130101  2013.0014     -18.0      76.8      -2.5      1.5      0.9       1.8
...
```

## 1) 时间滤波

脚本：`GNSS-coordinated-anomaly-filt.py`

作用：

- 把 `*_raw.neu` 读入并映射为**连续逐日序列**（缺测线性插值补齐，满足等间隔采样）
- 对 N/E/U（内部用 m）执行：
  - 分位裁剪 + 回插
  - EEMD
  - Hilbert 得瞬时频率
  - 在 20–150 天游程频带内逐日筛选 IMF 并求和得到滤波分量
- 输出每站结果与站点信息表

输出目录默认：`GNSS-coordinated-anomaly/FiltDataOut/`

输出文件：

- `<SITE>_HHTfilt.txt`
  - 表头：`YYYYMMDD  YYYY.DECM  N_filt(mm)  E_filt(mm)  U_filt(mm)  Azimuth(deg)`
  - 注意：N/E/U 输出为 **mm**（内部计算用 m）
- `stinfo.txt`
  - 表头：`site  lat  lon`
- `<SITE>_HHTfilt.png`（**可选**，默认生成）
  - 上 / 中 / 下三子图：N、E、U 的 HHT 滤波时间序列（mm）
  - 开关：脚本顶部 `DEFAULT_ENABLE_PLOT`（默认 `True`）；命令行 `--no-plot` 可关闭本次运行绘图

运行示例：

```bash
# 处理全部站点
python GNSS-coordinated-anomaly/GNSS-coordinated-anomaly-filt.py

# 指定时间范围（YYYYMMDD，含端点）
python GNSS-coordinated-anomaly/GNSS-coordinated-anomaly-filt.py --start-date 20130101 --end-date 20141231

# 只跑单站（用 glob）
python GNSS-coordinated-anomaly/GNSS-coordinated-anomaly-filt.py --glob-pattern "LS01_raw.neu" --max-jobs 1

# 不生成每站 PNG
python GNSS-coordinated-anomaly/GNSS-coordinated-anomaly-filt.py --no-plot
```

## 2) 空间格网统计与绘图

脚本：`GNSS-coordinated-anomaly-space.py`

读取 `FiltDataOut/` 下的：

- `stinfo.txt`
- `*_HHTfilt.txt`

并支持按日期筛选：

- **范围**：`--date-start YYYYMMDD --date-end YYYYMMDD`
- **指定点**：`--dates YYYYMMDD,YYYYMMDD,...`（优先级最高）

不带子命令直接运行时，可按脚本顶部默认配置执行 **grid + plot**（格网结果与图一般写入 `Frames/`；格网文本如 `ANGDIFF_MEAN.txt` 等，具体以脚本内 `DEFAULT_FRAMES_OUT` 为准）。

运行示例：

```bash
# 仅计算某一天的格网
python GNSS-coordinated-anomaly/GNSS-coordinated-anomaly-space.py grid --dates 20130101

# 画某个时间范围（逐日输出 PNG）
python GNSS-coordinated-anomaly/GNSS-coordinated-anomaly-space.py plot --date-start 20130101 --date-end 20130131
```

## 3) 站点对滑动相关系数

脚本：`GNSS-coordinated-anomaly-time.py`

- **输入**：`FiltDataOut/<SITE>_HHTfilt.txt`（与 filt 输出一致）
- **作用**：对给定站点对，在共同日期对齐后，按窗长 / 步长计算 N、E、U 的 **Pearson 相关系数**时间序列；时间戳取**窗尾** `YYYYMMDD`
- **输出目录**：`GNSS-coordinated-anomaly/TimeCorrelationOut/`（不存在则自动创建）
- **输出文件**：
  - `<SITE1>-<SITE2>-TimeCorrelation.txt`：`YYYYMMDD`、`N_correlation`、`E_correlation`、`U_correlation`
  - `<SITE1>-<SITE2>-TimeCorrelation.png`（**可选**，默认生成）：上 / 中 / 下三子图对应 N/E/U 相关系数
- **配置**：脚本顶部 `DEFAULT_SITE_PAIRS`、`DEFAULT_WINDOW_DAYS`、`DEFAULT_STEP_DAYS`、`DEFAULT_ENABLE_PLOT` 等；无命令行参数时直接使用这些默认值。`--no-plot` 关闭绘图。

运行示例：

```bash
# 直接运行（使用脚本内默认站点对与窗长）
python GNSS-coordinated-anomaly/GNSS-coordinated-anomaly-time.py

# 命令行指定站点对与窗长
python GNSS-coordinated-anomaly/GNSS-coordinated-anomaly-time.py --pair LS06 LS07 --window-days 30 --step-days 1
```

## 依赖安装

在 `GNSS-coordinated-anomaly/` 目录下：

```bash
pip install -r GNSS-coordinated-anomaly_requirements.txt
```

`matplotlib` 用于 **filt / space / time** 中的绘图；若仅需数值输出，可对 filt、time 使用 `--no-plot`，space 是否出图取决于你是否执行 plot 及脚本配置。