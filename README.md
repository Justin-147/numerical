# Numerical 程序说明

本目录包含地震相关观测数据的异常识别、预报效能评估及辅助工具等程序，主要面向单测项分析。各子目录与根目录程序简介如下。

---

## 根目录

### extract_columns.py

**多列数据按列提取。** 从多列文本文件中按指定列号抽取列并写入新文件。

- **功能**：支持有/无表头、中文或英文表头；列号从 1 开始；可选是否输出表头。
- **用途**：常与 `cycle-related-anomaly`、`trend-related-anomaly` 等输出的多列结果配合，抽取“时间 + 某特征列”供 R 值/Molchan 等程序使用。
- **运行**：修改文件内 `INPUT_FILE`、`OUTPUT_FILE`、`COLUMNS`、`HAS_HEADER`、`OUTPUT_HEADER` 等参数后执行 `python extract_columns.py`。

---

## 异常识别与特征提取

### cycle-related-anomaly（破年变异常）

**单测项破年变异常识别与预报效能评估。**

- **主程序**：`cycle_anomaly_single.py`
- **流程**：观测数据 → 预处理 → 年变显著性判断 → 傅里叶滑动年变拟合与异常提取 → 在「阈值×预报期」网格上计算 R 值 → 输出 Rmax、R0、报准/漏报统计及图件。
- **输入**：观测数据（时间+观测值）、多边形规则、地震目录、干扰信息（可选）。
- **输出**：`*_processed.txt`、`*_result.txt`、图件等。
- **说明**：详见 [cycle-related-anomaly/README.md](cycle-related-anomaly/README.md)。

### trend-related-anomaly（趋势转折异常）

**单测项趋势转折（速率变化）异常识别与预报效能评估。**

- **主程序**：`trend_anomaly_single.py`
- **流程**：观测数据 → 预处理、归一化 → 滑动窗口矢量转角极值提取 → 在「阈值×预报期」网格上计算 R 值 → 输出 Rmax、R0、报准/漏报统计及图件。
- **输入**：观测数据、多边形规则、地震目录、干扰信息（可选）。
- **输出**：`*_processed.txt`、`*_result.txt`、图件等。
- **说明**：详见 [trend-related-anomaly/README.md](trend-related-anomaly/README.md)。

### pdf-related-anomaly（概率密度拟合异常）

**基于概率密度拟合的异常分析（SXL 风格）。**

- **主程序**：`pdf_anomaly.py`
- **流程**：分钟采样观测 → 小波去趋势 → 滑动窗口内对标准化序列做经验 PDF + 理论 PDF 拟合得到 λ → 输出时序 λ² 及拟合误差，用于异常识别。
- **输入**：两列文本（时间码 + 观测值），缺测 99999/999999 自动剔除。
- **输出**：`*_detrend.txt`、`*_lmd.txt`（时间、λ²、误差）、图件（原始/去趋势/λ²）。
- **说明**：详见 [pdf-related-anomaly/README.md](pdf-related-anomaly/README.md)。

---

## 预报效能评估（基于已提取特征）

以下程序假定“异常/特征”已由前述流程或其它方式得到，数据文件为**两列（时间 + 数值）**，在此基础上做 R 值或 Molchan 图评估。

### R-value（R 值评估）

**基于 R 值的预报效能评估（单测项）。**

- **主程序**：`R_value_compute.py`
- **功能**：读入待评估数据文件（时间+数值）、地震目录、多边形规则、干扰信息，在「阈值×预报期」网格上计算 R 值，输出 Rmax、R0、报准/漏报统计及图件。
- **阈值模式**：可按“待评估数据倍数 STD”或“给定阈值范围”扫描。
- **说明**：详见 [R-value/README.md](R-value/README.md)。

### Molchan-graph（Molchan 图评估）

**基于 Molchan 图的预报效能评估（单测项）。**

- **主程序**：`Molchan_graph_compute.py`
- **功能**：在 R 值程序基础上，增加 Molchan 图（漏报率–占有率曲线、2.5% 显著性参考线、AUC、概率增益等），其余输入输出与 R-value 一致。
- **说明**：详见 [Molchan-graph/README.md](Molchan-graph/README.md)。

---

## 测试与示例

### pdf-test

**与 PDF 模型相关的合成数据与单窗绘图示例。**

- **gen_data_and_plot.py**：生成 3 组 30 天分钟采样数据（正态 X、对数正态 Y、乘积 Z=XY），绘制 6 子图（上排时序、下排 PDF）；另可输出 3 行×3 列、仅 PDF 的图（不同 λ）。用于理解尺度混合与 pdf_anomaly 中 λ 的对应关系。
- **plot_yasw_window_pdf.py**：从 `pdf-related-anomaly/yasw.txt` 截取指定起始时间后一个窗长的数据，做去趋势与标准化，拟合 λ，绘制该窗的经验 PDF（红圈）与理论 PDF（黑线），并标注 λ。依赖 `pdf_anomaly` 的 `load_data`、`filt_db`、`pdf_act`、`pdf_fit` 等。

---

## 基于GNSS异常基线、跨断层异常基线等判定异常断层段

本部分程序均位于 `fault-movement-anomaly/` 目录下。

### GNSS-baseline.py

**GNSS 基线长度与方位角分析。**

- **程序**：`fault-movement-anomaly/GNSS-baseline.py`
- **说明**：详见 [fault-movement-anomaly/GNSS-baseline_README.md](fault-movement-anomaly/GNSS-baseline_README.md)
- **依赖**：`fault-movement-anomaly/GNSS-baseline_requirements.txt`
- **要点**：计算 baseline(mm) 与 azimuth(millisecond) 的相对变化量与去趋势结果，并基于输入文件 `sig_n`、`sig_e` 做一阶误差传播（输出 `sigma_baseline(mm)`、`sigma_azimuth(millisecond)`）；方位角原始图包含 `Start Azimuth = xxxx.xxxx°` 标注。

### GNSS_baseline_fault_segment_intersection.py

**GNSS 异常基线 × 断层段线段相交判定。**

- **程序**：`fault-movement-anomaly/GNSS_baseline_fault_segment_intersection.py`
- **说明**：详见 [fault-movement-anomaly/GNSS-fault-intersection_README.md](fault-movement-anomaly/GNSS-fault-intersection_README.md)
- **依赖**：`fault-movement-anomaly/GNSS-fault-intersection_requirements.txt`
- **输入**：`FaultCord_justExample.xlsx` 与 `GNSS基线异常表-示例.xlsx`（固定表头；文件名在脚本顶部变量中修改）
- **输出**：`fault-movement-anomaly/Abnormal_Fault_Segments_from_GNSS_Baseline.txt`（同一基线可对应多条断层段，反之亦然；无相交也会生成仅表头文件）

### CrossFault-baseline.py

**跨断层基线或水准差分（不等间隔时间序列）。**

- **程序**：`fault-movement-anomaly/CrossFault-baseline.py`
- **说明**：详见 [fault-movement-anomaly/CrossFault-baseline_README.md](fault-movement-anomaly/CrossFault-baseline_README.md)
- **依赖**：`fault-movement-anomaly/CrossFault-baseline_requirements.txt`
- **输入/输出**：读取 `fault-movement-anomaly/CrossFault-Data/` 下两列数据（yyyymmdd + 数值），按月窗长计算差分，输出 `CrossFault-Out/` 下的差分 txt 与 PNG 图件。

### CrossFault-FaultAnomaly.py

**跨断层异常点 → 异常断层段（点到断层段距离筛选）。**

- **程序**：`fault-movement-anomaly/CrossFault-FaultAnomaly.py`
- **说明**：详见 [fault-movement-anomaly/CrossFault-FaultAnomaly_README.md](fault-movement-anomaly/CrossFault-FaultAnomaly_README.md)
- **依赖**：`fault-movement-anomaly/CrossFault-FaultAnomaly_requirements.txt`
- **输出**：`fault-movement-anomaly/Abnormal_Fault_Segments_from_CrossFault.txt`（无匹配则仅表头）

---

## 依赖与运行方式

- 各子目录一般提供 `requirements.txt`（或类似专用命名的依赖文件，例如 `fault-movement-anomaly/GNSS-baseline_requirements.txt`），在对应目录下执行 `pip install -r <依赖文件名>` 安装依赖。
- 程序入口多为各主程序文件末尾的 `if __name__ == "__main__":`，在脚本内修改数据路径、台站、震级、阈值、预报期等参数后运行，例如：
  ```bash
  cd cycle-related-anomaly && python cycle_anomaly_single.py
  cd R-value && python R_value_compute.py
  cd pdf-related-anomaly && python pdf_anomaly.py
  ```
- 建议 Python 3.8+（部分程序建议 3.9+）；可选安装 `cartopy` 以在空间分布图中使用底图。

---

## 目录结构概览

```text
numerical/
├── README.md                 # 本文件
├── extract_columns.py        # 多列提取
├── fault-movement-anomaly/   # 基于GNSS异常基线、跨断层异常基线等判定异常断层段
│   ├── GNSS-baseline.py      # GNSS 基线长度与方位角分析
│   ├── GNSS_baseline_fault_segment_intersection.py  # GNSS 异常基线 × 断层段线段相交判定
│   ├── CrossFault-baseline.py  # 跨断层基线或水准差分（不等间隔时间序列）
│   └── CrossFault-FaultAnomaly.py  # 跨断层异常点 → 异常断层段
├── R-value/                  # R 值评估
├── Molchan-graph/            # Molchan 图评估
├── cycle-related-anomaly/     # 破年变异常
├── trend-related-anomaly/     # 趋势转折异常
├── pdf-related-anomaly/       # 概率密度拟合异常
└── pdf-test/                 # PDF 合成数据与单窗绘图示例
```

