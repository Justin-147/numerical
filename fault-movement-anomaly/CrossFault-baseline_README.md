# 跨断层基线或水准差分（CrossFault-baseline）

本说明文档专门针对 `fault-movement-anomaly/CrossFault-baseline.py`。

该程序用于读取**两列**（时间 + 数值）的跨断层基线或水准数据，按给定“月窗长”在不等间隔时间序列上计算差分，并输出差分结果与图件（PNG）。

## 1. 主程序

- **主程序**：`CrossFault-baseline.py`
- **依赖安装**：

```bash
pip install -r CrossFault-baseline_requirements.txt
```

## 2. 输入输出

- **输入目录**：`CrossFault-Data/`
  - 示例：`CrossFault-Data/下关水2-水1.txt`
- **输出目录**：`CrossFault-Out/`（若不存在会自动创建）
  - 差分结果：`<文件名stem>_<窗长>月窗长差分.txt`
  - 图件：`<文件名stem>_<窗长>月窗长差分.png`

## 3. 输入数据格式

- 两列数据：第 1 列为 **8 位时间码** `yyyymmdd`；第 2 列为数值。
- 文件**可无表头**；分隔符支持：**逗号 / 空格 / tab**。
- 时间不等间隔：允许某些月份缺测、某些月份多次观测。

## 4. 差分逻辑（重点）

对每个计算点 A（时间 `tA`，值 `vA`），差分定义为：

- \(diff(tA) = vA - vB\)
- 输出时间取 **A 的时刻**（即 `tA`）

其中 B 的选择规则：

1. 令 `target` 为 `tA` 往前推 `WINDOW_MONTHS` 个月后的**年月**（不严格卡天）。
2. 若 `target` 这个年月没有任何观测，则该 A 点**不输出**，继续下一个 A。
3. 若 `target` 这个年月同月有多天观测，则选择“更接近窗长”的那个观测点：
   - 将候选点 `tB` 加上 `WINDOW_MONTHS` 个月得到 `tB'`
   - 选择使 \(|tA - tB'|\) 最小的 `tB` 参与计算

## 5. 可修改参数

在 `CrossFault-baseline.py` 顶部“可修改参数区”中修改：

- `WINDOW_MONTHS`：窗长（月），例如 3/6/12/24（默认 12）
- `FILE_GLOB`：默认处理 `CrossFault-Data/` 下全部 `*.txt`；也可改成某个具体文件名
- `DATA_DIR` / `OUT_DIR`：输入/输出目录

## 6. 绘图说明（PNG）

- 横轴：时间（label：`Date`）
  - 主 tick：每年 1 月 1 日；根据 x 轴范围自动稀疏，主 tick + ticklabel 不超过 10 个
  - 次 tick：每年 1 月 1 日（每年一个），不显示 ticklabel
- 纵轴：差分值（label：`mm`），范围为 \([min, max]\) 上下各扩 \(1/10\)
- 样式：青色空心小圆圈绘点，红色细线连接
- 字体：x/y 轴与刻度为 Times New Roman；title 允许中英文混排并自动回退到中文字体

