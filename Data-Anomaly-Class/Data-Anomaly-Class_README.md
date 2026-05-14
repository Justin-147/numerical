# Data-Anomaly-Class：多尺度 CNN-BiLSTM + 加性注意力时序分类

## 模型概述

基于**多尺度一维卷积 + 双层 BiLSTM + 加性注意力 + 两层全连接**的时序数据多分类模型。

注意：本程序只为跑通流程，并未用真实数据进行训练和验证，因此相关指标不具备参考意义，部分超参数还可以根据实际训练和验证效果进行调整。

- **多尺度 CNN**：三路并行 `Conv1D`，卷积核长度分别为 **3、5、7**（在 1D 序列上等价于沿时间维度的局部感受野），分别侧重短时突变、中程趋势与较长周期形态；每路后接 `BatchNormalization` 与 `MaxPool1D(2)`，再沿通道维拼接。
- **双层 BiLSTM**：在拼接后的时序特征上堆叠两层双向 LSTM，挖掘前向与反向的长距离依赖。
- **加性注意力**：对 BiLSTM 输出的每个时间步计算标量得分（经 `tanh` 的线性变换），再对整段序列做 **Softmax** 得到各步权重（非负且和为 1）。该权重即**注意力权重**：表示模型在汇总整条序列时「看多重视该时间步」；**不是**类别概率。按权重对隐状态加权求和得到**全局上下文向量**；训练结束后按**真实类别**对样本分组、组内对权重取平均并绘图；仅预测时按**预测类别**（含拒判 -1）分组绘图。
- **分类头**：两层 `Dense`（ReLU）+ `BatchNormalization` + `Dropout`，末层 `Dense(num_classes, softmax)`；卷积与全连接使用 **L2 正则**抑制过拟合。
- **训练**：**Adam** 优化器 + **sparse_categorical_crossentropy**（标签为整数 `0 .. num_classes-1`）。

## 文件说明


| 文件                                    | 说明                                                   |
| ------------------------------------- | ---------------------------------------------------- |
| `Data-Anomaly-Class.py`               | 模型定义、训练/仅预测；输入固定为**逐条序列**时间维 min-max 到 `[-1,1]`；未知拒判 |
| `Data-Anomaly-Class_requirements.txt` | Python 依赖                                            |
| `Data-Anomaly-Class_README.md`        | 本说明                                                  |


## 依赖安装

在 `Data-Anomaly-Class/` 目录下执行：

```bash
pip install -r Data-Anomaly-Class_requirements.txt
```

绘制网络结构图（`keras.utils.plot_model`）时，需安装 **pydot** 与 **graphviz**（本目录 `requirements` 已列出）。`pydot` 会调用系统中的 **dot** 可执行文件；若报错找不到 `dot`，请从 [Graphviz 官网](https://graphviz.org/download/) 安装 Graphviz 并把其 `bin` 目录加入系统 `PATH`（Windows 常见情况）。也可用 Conda：`conda install graphviz pydot`。若环境不完整，脚本会跳过结构图导出并打印提示。

### Windows：TensorFlow 报错「DLL load failed / 动态链接库初始化例程失败」

含义是 **TensorFlow 自带的 C++ 扩展未能加载**，与脚本逻辑无关。可按顺序尝试：

1. **与本仓库依赖一致**：`requirements` 已固定为 `**tensorflow==2.16.2`**。请先执行
  `pip install -r Data-Anomaly-Class_requirements.txt`  
   若曾装过 2.17+，建议先 `pip uninstall tensorflow tensorflow-intel` 再重装。
2. 安装 **Microsoft Visual C++ Redistributable 2015–2022（x64）**：
  [https://aka.ms/vs/17/release/vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe)  
   安装后重启终端再运行脚本。
3. 若 pip 仍失败，用 **独立 conda 环境**（避免污染 Anaconda base）：
  `conda create -n dac python=3.11 pip` → `conda activate dac` → 再 `pip install -r Data-Anomaly-Class_requirements.txt`。
4. 若 pip 提示 `**Ignoring invalid distribution ~ensorflow`**，说明 `site-packages` 里残留损坏目录；请删除 `Lib\site-packages` 下名称以 `~ensorflow` 或异常命名的 tensorflow 相关文件夹后重装。

## 运行

```bash
cd Data-Anomaly-Class

# 默认：训练（与脚本内 num_classes、序列长度等一致；当前示例为 7 类）
# 输入会自动对每条序列在时间维上做 [-1,1] min-max，无需额外参数或 norm 文件
python Data-Anomaly-Class.py

# 最大 epoch 与 EarlyStopping 耐心值（默认 50 / 10）
python Data-Anomaly-Class.py --epochs 80 --early-stop-patience 15

# 仅加载已训练权重推理（演示：仍用脚本内随机测试数据；真实使用请改为读入你的 .npy 等）
python Data-Anomaly-Class.py --mode predict --weights best_model.keras

# 调整「未知类」拒判阈值（softmax 最大值低于该值则输出类别 -1）
python Data-Anomaly-Class.py --mode predict --reject-threshold 0.55
```

默认在脚本所在目录生成（若运行成功）：

- `best_model.keras`：验证集 `val_loss` 最优时的权重
- `train_curve.png`：训练/验证损失与准确率随 **epoch** 变化
- `label_compare_train.png` / `label_compare_val.png` / `label_compare_test.png`：上图同轴散点（真标签 / 预测标签，图例区分）；下图各样本 **max softmax** 柱状图（训练子集 / 验证 / 测试）
- `confusion_matrix_train.png` / `confusion_matrix_val.png` / `confusion_matrix_test.png`：真实类别 × 预测类别混淆矩阵热力图（行归一化比例着色，格内为计数与比例）；最后一列为「预测为未知 (`-1`)」
- `confidence_hist_train.png` / `confidence_hist_val.png` / `confidence_hist_test.png`：全体样本的 **max softmax（置信度）** 直方图 + 按 **真实类别** 分组的置信度直方图
- `predict` 模式另存：`label_compare_predict_fulltest.png` 等同名前缀的混淆矩阵与置信度图（对脚本内整段测试集）
- `predict_label_compare_demo.png`：`predict` 模式下前若干条样本的对比图（演示）
- `attention_{train|val|test}_by_true_label_class{k}.png`：该 split 上**真实标签为 k** 的样本，其注意力权重在样本维上的**平均**曲线（无该类样本则不生成）
- `attention_predict_by_pred_label_class{k}.png` / `attention_predict_by_pred_label_reject.png`：`predict` 模式下按**预测标签**分组（拒判为 -1 时另存 `reject`）；无真值时用预测类归类
- `model_structure.png`：模型结构图（需 pydot + Graphviz）

## 输入缩放（逐条序列 `[-1,1]`）

本脚本**仅**采用：对形状 `(N, T, C)` 的每条样本，在**时间维 `T`** 上对该条数据求 min/max，线性映射到 `[-1,1]` 并裁剪。训练、验证、测试与 `predict` 模式使用**同一规则**，**不生成、不读取**任何 `norm_bounds` 类文件。该做法弱化跨样本的绝对幅值，更侧重波形形态；若你已在管线中做过缩放，可在 `main()` 中去掉对 `apply_minmax_neg1_pos1_per_sequence` 的调用以免二次缩放。

## 训练固定 K 类（如 7 类），但实际会遇到「非这 K 类」怎么办

1. **推理拒判（脚本已实现启发式）**：根据 softmax **最大概率**是否低于 `--reject-threshold`，输出 **-1 表示未知**。阈值需在验证集上标定；缺点是对「类内难例」也可能判未知。
2. **增加第 K+1 类「其它/未知」**：收集难以归入现有 7 类的样本标为第 8 类，`num_classes` 改为 8 重新训练（需真实标签或伪标签策略）。
3. **开放集方法**（需改模型/损失）：如能量分数、ProtoNet、OOD detector 等，超出本示例范围。

**训练阶段**：标签必须在 `0 .. num_classes-1`；若出现越界整数，`sparse_categorical_crossentropy` 会异常，应在数据管道中过滤或重映射。

## 接入真实数据

1. 将 `main()` 中的 `x_train` / `y_train` 等替换为你的数组：`x` 形状 `(N, T, C)`，`T` 与 `input_shape` 一致，`C` 为通道数；`y` 形状 `(N,)`，整型类别下标 `0 .. num_classes-1`。
2. 按需修改 `main()` 内 `input_shape`、`num_classes`，并与 `build_model(..., num_classes=K)` 保持一致。

## 调参原则（简要）

- 先看 **训练集 vs 验证集** 的 loss/acc 差距：差距大 → 略增大 **FC 与 concat 处** Dropout，或加强 **L2**、或加 **LSTM dropout**；差距小甚至欠拟合 → 应**减小** Dropout 或加深/加宽网络。
- **BatchNorm 与 Dropout 同时使用**时，不宜把 Dropout 设得过大叠在 BN 前同一位置反复“随机化”，否则等效学习率波动大；本模型顺序为 **Conv → BN → Pool**、**Dense → BN → Dropout**，是较常见写法。
- 样本很少时，整体 Dropout 宜**略降**（例如 concat 0.2、FC 0.25/0.2），否则欠拟合风险上升。

