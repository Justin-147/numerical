# 单测项破年变预报效能分析

对单个测项观测序列做破年变异常识别与预报效能评估：读取观测数据与地震目录，经预处理、年变显著性判断、傅里叶滑动年变拟合与异常提取，在「阈值×预测期」网格上计算 R 值，输出 Rmax、R0、报准/漏报统计及图件。

## 1. 程序与依赖

- **主程序**：`cycle_anomaly_single.py`（单测项完整流程）
- **依赖安装**（建议 Python 3.9+）：

```bash
pip install -r requirements.txt
```

- **可选**：若需在空间分布图中使用 **cartopy** 底图，在程序中设置 `USE_CARTOPY_MAP = True` 并安装：`pip install cartopy`。

## 2. 输入文件

| 文件 | 说明 |
|------|------|
| **观测数据** | 文本，两列：`时间标签`、`观测值`。时间可为 `yyyymmddhh`（小时值）或 `yyyymmdd`（日值）。 |
| **多边形规则** | 文本，首行为标题（可含中文），其后两列：`经度`、`纬度`，每行一个顶点，首尾闭合。 |
| **地震目录** | EQT 定长格式，每行包含年月日时分秒、经纬度、震级、深度等。 |
| **干扰信息**（可选） | 文本文件。首行为标题，前两列：干扰开始时间、结束时间（时间码格式同观测数据）。不存在或为空时按无干扰处理。 |

观测数据、多边形、地震目录**内容为空**时程序会报错并退出；干扰文件可为空或不存在。

## 3. 配置与运行

在 `cycle_anomaly_single.py` 末尾 `if __name__ == "__main__":` 中修改以下参数后运行：

```python
DATA_FILE = Path(r"你的观测数据路径.txt")
POLYGON_FILE = Path(r"你的多边形规则路径.txt")
EQ_CATALOG_FILE = Path(r"你的地震目录路径.eqt")
INTERFERENCE_FILE = Path(r"干扰信息路径.txt")   # 无则填 None
STATION_LON = 台站经度
STATION_LAT = 台站纬度
MAG_MIN = 预测震级下限      # 震级下限
MAG_MAX = 预测震级上限      # 震级上限
RATE_MIN = 0.0              # 阈值倍数最小值
RATE_MAX = 10.0             # 阈值倍数最大值
RATE_STEP = 0.1             # 阈值倍数步长
ALM_DAY_START = 60          # 预报期起点（天）
ALM_DAY_END = 720           # 预报期终点（天）
ALM_DAY_STEP = 1            # 预报期步长（天）
USE_CARTOPY_MAP = False     # True 时需安装 cartopy
```

运行：

```bash
python cycle_anomaly_single.py
```

## 4. 流程概览

1. 读取观测数据；缺数插值（PCHIP）、台阶修正；若为小时值则转日均值。
2. 小波分解，得到年变分析用序列（去近似 + 去 1–4 阶细节）及绘图用原始年变序列。
3. 年变显著性判断（FFT + F 检验与主峰判据）；不显著则只写结果说明并退出。
4. 三年滑动傅里叶年变拟合，得到残差序列与标准差。
5. 输出处理序列文件：`<数据文件名>_processed.txt`（Time, Filtered Annual, Fitted Annual, Residual, Anomaly）。
6. 按多边形、预测震级范围、观测时间范围等从地震目录中筛选震例；读取干扰时段（若有）。
7. 在 (阈值×预测期) 网格上计算 R 矩阵，取 Rmax 及对应阈值、预测期；计算 R0 与报准/漏报。
8. 写结果文件 `<数据文件名>_result.txt`，并生成图件 `<数据文件名>_result.png`。

## 5. 输出文件

- **`<数据文件名>_result.txt`**：破年变异常与预报效能结果（Rmax、R0、最优阈值与预报期、异常起止、报准/漏报及震例信息等）。
- **`<数据文件名>_processed.txt`**：五列，分别为 Time（YYYYMMDD）、Filtered_Annual（去趋势并经小波平滑后的年变成分）、Fitted_Annual（傅里叶拟合年变）、Residual（残差）、Anomaly（满足最优阈值且非干扰时段为 1，否则为 0）。
- **`<数据文件名>_result.png`**：四幅子图（年变曲线、空间分布、残差与异常/预警/震例、R–TT 曲面）。

所有输出文件与观测数据文件同目录。
