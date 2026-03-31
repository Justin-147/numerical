# 跨断层异常点 → 异常断层段（CrossFault-FaultAnomaly）

首先针对跨断层基线或水准观测数据，利用CrossFault-baseline.py程序计算获得差分结果（差分方法目前对于跨断层基线或水准观测数据效果相对较好），基于专家经验结合异常核实等情况判定哪些跨断层基线或水准测项为有预测意义的异常测项。
然后将跨断层异常基线或水准的测项信息填入跨断层异常表，同时需要提供各断层段的坐标文件。
最后运行本程序，即可输出异常断层段的信息（判定逻辑为断层段附近存在跨断层基线或水准异常）。

本说明文档专门针对 `fault-movement-anomaly/CrossFault-FaultAnomaly.py`。

该程序用于读取跨断层异常场地（点）与断层段迹线（折线），计算异常点到各断层段的最小距离，并在 5 km 内选取最近断层段作为“异常断层段”，输出结果表。

## 1. 主程序

- **主程序**：`CrossFault-FaultAnomaly.py`
- **依赖安装**：

```bash
pip install -r CrossFault-FaultAnomaly_requirements.txt
```

## 2. 输入文件（固定表头）

默认在 `fault-movement-anomaly/` 目录下放置：

- `跨断层异常表-示例.xlsx`（跨断层异常表）
  - 固定表头：`场地名称`、`手段名称`、`经度`、`纬度`
- `FaultCord_justExample.xlsx`（断层段迹线坐标）
  - 固定表头：`断层编号`、`断层段编号`、`断层名称`、`断层段名称`、`经度`、`纬度`

输入/输出文件名与阈值在脚本顶部“可修改参数区”中修改：

- `ABNORMAL_XLSX`
- `FAULT_XLSX`
- `OUTPUT_TXT`
- `MAX_DIST_KM`（默认 5 km）

## 3. 判定逻辑

对每个异常场地（点）：

1. 将每条断层段迹线视为折线（相邻点构成线段）；
2. 计算异常点到该折线上所有线段的最小距离（局部平面近似，单位 km）；
3. 选择最近的断层段；若其距离 \(\le\) `MAX_DIST_KM`，则输出该异常点对应的异常断层段。

若某异常点在阈值范围内找不到任何断层段，则该点不输出，但输出文件仍会生成（仅表头）。

## 4. 输出

输出文件：`Abnormal_Fault_Segments_from_CrossFault.txt`（tab 分隔，含表头）

字段：

- `fault_id`
- `segment_id`
- `fault_name`
- `segment_name`
- `abnormal_site`
- `abnormal_item`
- `lon_site`
- `lat_site`

其中 `lon_site`、`lat_site` 在输出中保留 **2 位小数**。

## 5. 运行方式

在 `numerical/fault-movement-anomaly` 目录下运行：

```bash
cd fault-movement-anomaly
python CrossFault-FaultAnomaly.py
```

