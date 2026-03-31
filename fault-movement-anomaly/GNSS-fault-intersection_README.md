# 基于GNSS 基线异常判断异常断层段（GNSS-fault-intersection）

首先针对GNSS各站点原始位移时序数据，利用GNSS-baseline.py程序获得不同GNSS基线的时序结果，基于专家经验结合异常核实等情况判定哪些GNSS基线为有预测意义的异常基线。
然后将GNSS异常基线站点对信息填入GNSS基线异常表，同时需要提供各断层段的坐标文件。
最后运行本程序，即可输出异常断层段的信息（判定逻辑为GNSS异常基线跨过的断层段）。


本说明文档专门针对本目录中的 `GNSS_baseline_fault_segment_intersection.py` 程序，用于：

- 读取断层段迹线点位（折线）与 GNSS 基线异常站点对；
- 判断每条异常 GNSS 基线（两站连线）与哪些断层段折线**线段相交**（不考虑延长线）；
- 输出“异常断层段—异常基线对”的对应关系表。

## 1. 主程序

- **主程序**：`GNSS_baseline_fault_segment_intersection.py`
- **依赖安装**：

```bash
pip install -r GNSS-fault-intersection_requirements.txt
```

## 2. 输入文件

默认在 `fault-movement-anomaly/` 目录下放置：

- `FaultCord_justExample.xlsx`：断层段迹线点位（示例文件，**固定表头**）
  - 每条断层段是一条折线，由多行点位坐标构成；
  - 程序会按（断层编号 + 断层段编号）分组，将同组点按表中顺序连接为折线。
- `GNSS基线异常表-示例.xlsx`：GNSS 基线异常表（示例文件，**固定表头**）
  - 必须包含站点名称与经纬度。

### 固定表头要求

- **断层迹线表**必须包含以下列名：
  - `断层编号`、`断层段编号`、`断层名称`、`断层段名称`、`经度`、`纬度`
- **GNSS 异常表**必须包含以下列名：
  - `站点1名称`、`站点1经度`、`站点1纬度`、`站点2名称`、`站点2经度`、`站点2纬度`

## 3. 判定逻辑

- 将异常 GNSS 基线视为线段 AB（A、B 为两站经纬度点）。
- 将断层段迹线视为折线，每相邻两点构成一条线段 CD。
- 若 AB 与任一 CD 发生线段相交（含端点接触与共线重叠），则认为该基线与该断层段相交。
- 同一条基线可以与多条断层段相交；同一条断层段也可以与多条基线相交，输出会按行展开。

## 4. 输出

程序会生成：

- `Abnormal_Fault_Segments_from_GNSS_Baseline.txt`

字段为：

- `fault_id`
- `segment_id`
- `fault_name`
- `segment_name`
- `abnormal_gnss_baseline_pair`（如 `YNYA-XIAG`）

若没有任何相交记录，输出文件也会生成（仅表头）。

## 5. 运行方式

在 `numerical/fault-movement-anomaly` 目录下运行：

```bash
cd fault-movement-anomaly
python GNSS_baseline_fault_segment_intersection.py
```

输入/输出文件名在脚本顶部变量中修改：

- `FAULT_XLSX`
- `ABNORMAL_XLSX`
- `OUTPUT_TXT`