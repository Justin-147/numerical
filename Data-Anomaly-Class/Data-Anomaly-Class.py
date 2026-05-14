# 基于多尺度 CNN-BiLSTM 与加性注意力机制的时序数据分类模型
# =============================================================================
# 结构：3 路并行 Conv1D（核 3/5/7）→ 拼接 → 双层 BiLSTM → 加性注意力 →
#       两层全连接（BN + Dropout + L2）→ Softmax 多分类
# 优化：Adam + sparse_categorical_crossentropy（整数标签）
# =============================================================================
from __future__ import annotations

import argparse
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2


# -----------------------------------------------------------------------------
# 1. 加性注意力（Bahdanau 风格：score = v^T tanh(W h)）
# -----------------------------------------------------------------------------
class AdditiveAttention(keras.layers.Layer):
    def __init__(self, attn_units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.attn_units = int(attn_units)
        self.W = layers.Dense(self.attn_units, activation="tanh", name="attn_dense")
        self.v = layers.Dense(1, name="attn_score")

    def call(self, inputs):
        # inputs: (batch, time, dim)
        x = self.W(inputs)
        e = self.v(x)  # (batch, time, 1)
        alpha = tf.nn.softmax(e, axis=1)
        context = tf.reduce_sum(alpha * inputs, axis=1)  # (batch, dim)
        return context, tf.squeeze(alpha, axis=-1)  # alpha: (batch, time)

    def get_config(self):
        config = super().get_config()
        config.update({"attn_units": self.attn_units})
        return config


# -----------------------------------------------------------------------------
# 2. 构建模型（多输出：prediction + attention，便于训练与可视化）
# -----------------------------------------------------------------------------
def build_model(
    input_shape: tuple[int, int] = (100, 1),
    num_classes: int = 5,
    l2_reg: float = 1e-4,
) -> Model:
    # Dropout 放置与取值说明见 Data-Anomaly-Class_README.md「Dropout 放在哪些环节…」
    drop_after_concat = 0.3
    drop_fc_64 = 0.4
    drop_fc_32 = 0.3

    inputs = layers.Input(shape=input_shape, name="input")

    # 多尺度 CNN（3 / 5 / 7）
    b1 = layers.Conv1D(
        32, 3, padding="same", activation="relu", kernel_regularizer=l2(l2_reg), name="conv1d_3"
    )(inputs)
    b1 = layers.BatchNormalization(name="bn_cnn_3")(b1)
    b1 = layers.MaxPool1D(2, name="pool_cnn_3")(b1)

    b2 = layers.Conv1D(
        32, 5, padding="same", activation="relu", kernel_regularizer=l2(l2_reg), name="conv1d_5"
    )(inputs)
    b2 = layers.BatchNormalization(name="bn_cnn_5")(b2)
    b2 = layers.MaxPool1D(2, name="pool_cnn_5")(b2)

    b3 = layers.Conv1D(
        32, 7, padding="same", activation="relu", kernel_regularizer=l2(l2_reg), name="conv1d_7"
    )(inputs)
    b3 = layers.BatchNormalization(name="bn_cnn_7")(b3)
    b3 = layers.MaxPool1D(2, name="pool_cnn_7")(b3)

    concat = layers.concatenate([b1, b2, b3], name="concat_multiscale")
    concat = layers.Dropout(drop_after_concat, name="drop_concat")(concat)

    # 双层 BiLSTM
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True), name="bilstm_1")(concat)
    x = layers.BatchNormalization(name="bn_bilstm_1")(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True), name="bilstm_2")(x)
    x = layers.BatchNormalization(name="bn_bilstm_2")(x)

    context, attn_weights = AdditiveAttention(64, name="additive_attention")(x)

    x = layers.Dense(64, activation="relu", kernel_regularizer=l2(l2_reg), name="fc_64")(context)
    x = layers.BatchNormalization(name="bn_fc_64")(x)
    x = layers.Dropout(drop_fc_64, name="drop_fc_64")(x)

    x = layers.Dense(32, activation="relu", kernel_regularizer=l2(l2_reg), name="fc_32")(x)
    x = layers.BatchNormalization(name="bn_fc_32")(x)
    x = layers.Dropout(drop_fc_32, name="drop_fc_32")(x)

    prediction = layers.Dense(num_classes, activation="softmax", name="prediction")(x)

    return Model(
        inputs=inputs,
        outputs={"prediction": prediction, "attention": attn_weights},
        name="Multiscale_CNN_BiLSTM_AdditiveAttention",
    )


def attention_supervision_time_steps(input_shape: tuple[int, int]) -> int:
    """与 build_model 中每路 Conv 后单次 MaxPool1D(2) 一致：T -> T/2。"""
    return int(input_shape[0]) // 2


def supervision_dict_for_fit(y_cls: np.ndarray, input_shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """
    Keras 3 多输出模型在 fit/evaluate 时要求为每个命名输出提供 y。
    attention 不参与监督，仅占位形状；损失见 _neutral_attention_loss。
    """
    y_cls = np.asarray(y_cls, dtype=np.int32)
    n = int(y_cls.shape[0])
    t = attention_supervision_time_steps(input_shape)
    return {
        "prediction": y_cls,
        "attention": np.zeros((n, t), dtype=np.float32),
    }


class _NeutralAttentionLoss(keras.losses.Loss):
    """对 attention 输出恒为 0 的标量损失，不产生梯度，避免误训练注意力权重。"""

    def call(self, y_true, y_pred):
        del y_true
        return tf.cast(0.0, dtype=y_pred.dtype)


def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def apply_minmax_neg1_pos1_per_sequence(x: np.ndarray) -> np.ndarray:
    """
    对每条样本在「时间步」维上独立做 min-max 到 [-1,1]（形状 (N,T,C)）。
    训练、验证、测试与预测均使用同一规则；不读写任何额外缩放参数文件。
    注意：会削弱「跨样本绝对幅值」信息，更适合形态/相对变化分类。
    """
    x = np.asarray(x, dtype=np.float32)
    vmin = np.min(x, axis=1, keepdims=True)  # (N,1,C)
    vmax = np.max(x, axis=1, keepdims=True)
    rng = np.maximum(vmax - vmin, np.float32(1e-8))
    y = 2.0 * (x - vmin) / rng - 1.0
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def plot_true_pred_pmax(
    out_png: str,
    title: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pmax: np.ndarray,
    *,
    max_points: int = 400,
) -> None:
    """上图：真标签与预测标签同轴散点（图例区分）；下图：max softmax 概率柱状图。"""
    n = min(int(y_true.shape[0]), int(y_pred.shape[0]), int(pmax.shape[0]), max_points)
    idx = np.arange(n, dtype=np.float64)
    yt = np.asarray(y_true[:n]).ravel()
    yp = np.asarray(y_pred[:n]).ravel()
    pm = np.asarray(pmax[:n], dtype=np.float64).ravel()

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    ax0, ax1 = axes[0], axes[1]
    offset = 0.18
    ax0.scatter(idx - offset, yt, s=14, c="tab:blue", alpha=0.75, label="true label", edgecolors="none")
    ax0.scatter(
        idx + offset,
        yp,
        s=14,
        c="tab:orange",
        alpha=0.75,
        label="pred label (-1=unknown)",
        edgecolors="none",
    )
    ax0.set_ylabel("class")
    ax0.legend(loc="upper right", fontsize=9)
    ax0.grid(True, alpha=0.3)

    w = 0.85 if n <= 80 else max(0.02, 0.85 * 80.0 / float(n))
    ax1.bar(idx, pm, width=w, color="tab:green", alpha=0.75, edgecolor="white", linewidth=0.3, label="max softmax prob")
    ax1.set_ylabel("max softmax prob")
    ax1.set_xlabel("sample index (subset)")
    ax1.set_ylim(0.0, 1.0)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def confusion_matrix_true_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    行：真实类别 0..K-1；列：预测类别 0..K-1 与「预测为未知」一列（pred=-1 计入最后一列）。
    """
    yt = np.asarray(y_true, dtype=np.int64).ravel()
    yp = np.asarray(y_pred, dtype=np.int64).ravel()
    yp_m = np.where(yp < 0, num_classes, yp)
    yp_m = np.clip(yp_m, 0, num_classes)
    valid = (yt >= 0) & (yt < num_classes)
    yt = yt[valid]
    yp_m = yp_m[valid]
    cm = np.zeros((num_classes, num_classes + 1), dtype=np.int64)
    if yt.size:
        np.add.at(cm, (yt, yp_m), 1)
    return cm


def plot_confusion_matrix_heatmap(
    out_png: str,
    title: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> None:
    cm = confusion_matrix_true_pred(y_true, y_pred, num_classes)
    # 归一化到行（按真实类）比例，便于看「分错到哪」；同时保留计数标注
    row_sum = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_frac = cm.astype(np.float64) / row_sum

    fig, ax = plt.subplots(figsize=(max(8.0, num_classes * 0.9), 6.5))
    im = ax.imshow(cm_frac, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("fraction of true class (row-normalized)")

    col_labels = [str(i) for i in range(num_classes)] + ["unk"]
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_yticklabels([str(i) for i in range(num_classes)])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{int(cm[i, j])}\n({cm_frac[i, j]:.2f})",
                ha="center",
                va="center",
                color="white" if cm_frac[i, j] > 0.55 else "black",
                fontsize=7,
            )

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_confidence_histograms(
    out_png: str,
    title: str,
    y_true: np.ndarray,
    pmax: np.ndarray,
    num_classes: int,
) -> None:
    """
    上图：全体样本 softmax 最大值（预测置信度）直方图。
    下图：按「真实类别」分组的 p_max 直方图（子图），观察各类上的置信度分布。
    """
    yt = np.asarray(y_true, dtype=np.int64).ravel()
    pm = np.asarray(pmax, dtype=np.float64).ravel()
    n = min(yt.shape[0], pm.shape[0])
    yt, pm = yt[:n], pm[:n]
    mask = (yt >= 0) & (yt < num_classes)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.6], hspace=0.28)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.hist(pm[mask], bins=40, range=(0.0, 1.0), color="tab:blue", alpha=0.85, edgecolor="white")
    ax0.set_xlabel("max softmax prob (confidence)")
    ax0.set_ylabel("count")
    ax0.set_title("Overall prediction confidence")
    ax0.grid(True, alpha=0.3)

    ncols = int(np.ceil(np.sqrt(num_classes)))
    nrows = int(np.ceil(num_classes / ncols))
    gs_inner = gs[1, 0].subgridspec(nrows, ncols, wspace=0.35, hspace=0.45)
    for c in range(num_classes):
        r, col = divmod(c, ncols)
        ax = fig.add_subplot(gs_inner[r, col])
        sel = mask & (yt == c)
        if np.any(sel):
            ax.hist(pm[sel], bins=25, range=(0.0, 1.0), color="tab:orange", alpha=0.85, edgecolor="white")
        ax.set_title(f"true class {c} (n={int(np.sum(sel))})")
        ax.set_xlim(0.0, 1.0)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def mean_attention_for_subset(
    model: Model,
    x: np.ndarray,
    mask: np.ndarray,
    *,
    batch_size: int = 64,
) -> np.ndarray | None:
    """对 mask 选中的样本分批 predict，返回注意力在时间维上的样本平均 (T,)。"""
    mask = np.asarray(mask, dtype=bool).ravel()
    if mask.shape[0] != len(x):
        raise ValueError("mask 长度须与 x 的样本维一致")
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    parts: list[np.ndarray] = []
    for start in range(0, idx.size, batch_size):
        sel = idx[start : start + batch_size]
        out = model.predict(x[sel], batch_size=min(batch_size, len(sel)), verbose=0)
        parts.append(np.asarray(out["attention"], dtype=np.float64))
    attn = np.concatenate(parts, axis=0)
    return np.mean(attn, axis=0)


def predict_labels_batch(
    model: Model,
    x: np.ndarray,
    *,
    reject_threshold: float,
    batch_size: int = 64,
) -> np.ndarray:
    """全量推理整数标签（含拒判 -1）。"""
    labs: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        out = model.predict(x[start : start + batch_size], batch_size=batch_size, verbose=0)
        probs = np.asarray(out["prediction"])
        lab, _ = predict_with_unknown_reject(probs, confidence_threshold=reject_threshold)
        labs.append(np.asarray(lab))
    return np.concatenate(labs, axis=0)


def _plot_mean_attention_figure(mean_w: np.ndarray, title: str, out_png: str, *, dpi: int = 300) -> None:
    mean_w = np.asarray(mean_w, dtype=np.float64).ravel()
    t = np.arange(len(mean_w), dtype=np.int32)
    plt.figure(figsize=(10, 3))
    plt.plot(t, mean_w, color="#2E86AB")
    plt.fill_between(t, mean_w, alpha=0.3, color="#2E86AB")
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Mean attention weight")
    ymax = float(np.max(mean_w)) * 1.12 if mean_w.size else 1.0
    plt.ylim(0.0, max(ymax, 1e-8))
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()


def save_attention_mean_by_label_groups(
    model: Model,
    x: np.ndarray,
    labels: np.ndarray,
    out_dir: str,
    split: str,
    *,
    group_by: str,
    num_classes: int,
    batch_size: int = 64,
) -> int:
    """
    按标签分组，每组内对注意力权重在样本维上取平均，每个非空组单独保存一张图。
    group_by='true'：用真实标签（train/val/test）；group_by='pred'：用预测标签（无真值时的推理场景）。
    预测为 -1 时另存 attention_{split}_by_pred_label_reject.png。
    """
    if group_by not in ("true", "pred"):
        raise ValueError("group_by 须为 'true' 或 'pred'")
    labels = np.asarray(labels).ravel()
    if labels.shape[0] != len(x):
        raise ValueError("labels 与 x 样本数不一致")

    by = "true_label" if group_by == "true" else "pred_label"
    n_saved = 0
    for c in range(num_classes):
        mask = labels == c
        if not np.any(mask):
            continue
        mean_w = mean_attention_for_subset(model, x, mask, batch_size=batch_size)
        if mean_w is None:
            continue
        n = int(np.sum(mask))
        title = f"Mean attention ({split}, {group_by} label={c}, n={n})"
        out_png = os.path.join(out_dir, f"attention_{split}_by_{by}_class{c}.png")
        _plot_mean_attention_figure(mean_w, title, out_png)
        n_saved += 1

    if group_by == "pred":
        mask = labels == -1
        if np.any(mask):
            mean_w = mean_attention_for_subset(model, x, mask, batch_size=batch_size)
            if mean_w is not None:
                n = int(np.sum(mask))
                title = f"Mean attention ({split}, pred=reject/unknown, n={n})"
                out_png = os.path.join(out_dir, f"attention_{split}_by_pred_label_reject.png")
                _plot_mean_attention_figure(mean_w, title, out_png)
                n_saved += 1

    return n_saved


def predict_with_unknown_reject(
    probs: np.ndarray,
    *,
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    若 max(softmax) < confidence_threshold，则判为「未知」(-1)。
    用于训练类数固定、但推理时可能存在分布外/未建模类别的场景（启发式，阈值需自行标定）。
    """
    probs = np.asarray(probs, dtype=np.float64)
    pmax = np.max(probs, axis=-1)
    cls = np.argmax(probs, axis=-1).astype(np.int32)
    unk = pmax < float(confidence_threshold)
    out = cls.copy()
    out[unk] = -1
    return out, pmax.astype(np.float32)


def load_trained_model(weights_path: str) -> Model:
    return keras.models.load_model(
        weights_path,
        custom_objects={
            "AdditiveAttention": AdditiveAttention,
            "_NeutralAttentionLoss": _NeutralAttentionLoss,
        },
        compile=False,
    )


def save_eval_plots_for_split(
    model: Model,
    x: np.ndarray,
    y: np.ndarray,
    out_dir: str,
    split: str,
    *,
    num_classes: int,
    reject_threshold: float,
    max_points: int = 400,
) -> None:
    """一次前向：散点对比 + 混淆矩阵热力图 + 置信度直方图。"""
    out = model.predict(x, batch_size=64, verbose=0)
    probs = np.asarray(out["prediction"])
    pred, pmax = predict_with_unknown_reject(probs, confidence_threshold=reject_threshold)
    plot_true_pred_pmax(
        os.path.join(out_dir, f"label_compare_{split}.png"),
        f"{split}: true vs pred vs max softmax prob",
        y,
        pred,
        pmax,
        max_points=max_points,
    )
    plot_confusion_matrix_heatmap(
        os.path.join(out_dir, f"confusion_matrix_{split}.png"),
        f"Confusion matrix (row-normalized): {split}",
        y,
        pred,
        num_classes,
    )
    plot_confidence_histograms(
        os.path.join(out_dir, f"confidence_hist_{split}.png"),
        f"Confidence (max softmax) histograms: {split}",
        y,
        pmax,
        num_classes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="多尺度 CNN-BiLSTM + 加性注意力时序分类")
    parser.add_argument(
        "--mode",
        choices=("train", "predict"),
        default="train",
        help="train：训练并保存权重；predict：仅加载权重做推理演示",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="predict 模式下权重文件路径，默认 <脚本目录>/best_model.keras",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="最大训练轮次（epoch），默认 50；可与 EarlyStopping 提前结束配合",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=10,
        help="val_loss 无改善时最多再等多少个 epoch 后提前结束，默认 10",
    )
    parser.add_argument(
        "--reject-threshold",
        type=float,
        default=0.45,
        help="推理时若 softmax 最大值低于该阈值则输出类别 -1（未知），默认 0.45",
    )
    args = parser.parse_args()

    out_dir = _script_dir()
    # 以下为你已调整的参数（保持）
    input_shape = (100, 1)  # 单个样本的时间步长为100，通道数为1
    num_classes = 7  # 7个类别

    ckpt_path = os.path.join(out_dir, "best_model.keras")
    weights_path = args.weights or ckpt_path

    rng = np.random.default_rng(0)
    x_train = rng.standard_normal((2000, input_shape[0], input_shape[1]), dtype=np.float32)
    y_train = rng.integers(0, num_classes, size=2000, dtype=np.int32)
    x_val = rng.standard_normal((400, input_shape[0], input_shape[1]), dtype=np.float32)
    y_val = rng.integers(0, num_classes, size=400, dtype=np.int32)
    x_test = rng.standard_normal((400, input_shape[0], input_shape[1]), dtype=np.float32)
    y_test = rng.integers(0, num_classes, size=400, dtype=np.int32)

    if args.mode == "predict":
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"未找到权重文件：{weights_path}（请先 train 或指定 --weights）")
        model = load_trained_model(weights_path)
        x_test_proc = apply_minmax_neg1_pos1_per_sequence(x_test)

        batch = x_test_proc[:16]
        out = model.predict(batch, verbose=0)
        probs = np.asarray(out["prediction"])
        labels, pmax = predict_with_unknown_reject(probs, confidence_threshold=args.reject_threshold)
        print("predict 模式：前 16 条样本的类别（-1 表示未知）与最大概率：")
        for i in range(len(labels)):
            print(f"  [{i}] class={int(labels[i])}, p_max={float(pmax[i]):.4f}")
        plot_true_pred_pmax(
            os.path.join(out_dir, "predict_label_compare_demo.png"),
            "Predict demo (first 16): true vs pred vs max prob",
            y_test[:16],
            labels,
            pmax,
            max_points=16,
        )
        save_eval_plots_for_split(
            model,
            x_test_proc,
            y_test,
            out_dir,
            "predict_fulltest",
            num_classes=num_classes,
            reject_threshold=float(args.reject_threshold),
            max_points=400,
        )
        pred_all = predict_labels_batch(
            model,
            x_test_proc,
            reject_threshold=float(args.reject_threshold),
        )
        n_att = save_attention_mean_by_label_groups(
            model,
            x_test_proc,
            pred_all,
            out_dir,
            "predict",
            group_by="pred",
            num_classes=num_classes,
        )
        print(
            f"已保存预测对比图：{os.path.join(out_dir, 'predict_label_compare_demo.png')}；"
            f"全量测试集：confusion_matrix_predict_fulltest.png / confidence_hist_predict_fulltest.png；"
            f"按预测类平均注意力图 {n_att} 张（attention_predict_by_pred_label_*.png）"
        )
        return

    # ---------- train：逐条序列在时间维上缩放到 [-1,1] ----------
    x_train = apply_minmax_neg1_pos1_per_sequence(x_train)
    x_val = apply_minmax_neg1_pos1_per_sequence(x_val)
    x_test = apply_minmax_neg1_pos1_per_sequence(x_test)
    print("已对 train / val / test 逐条做 [-1,1] 归一化（每条序列独立，不保存缩放参数文件）")

    model = build_model(input_shape=input_shape, num_classes=num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            "prediction": "sparse_categorical_crossentropy",
            "attention": _NeutralAttentionLoss(),
        },
        metrics={"prediction": ["accuracy"]},
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=int(args.early_stop_patience),
            restore_best_weights=True,
        ),
        ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        x_train,
        supervision_dict_for_fit(y_train, input_shape),
        validation_data=(x_val, supervision_dict_for_fit(y_val, input_shape)),
        batch_size=32,
        epochs=int(args.epochs),
        callbacks=callbacks,
        verbose=1,
    )

    try:
        test_metrics = model.evaluate(
            x_test,
            supervision_dict_for_fit(y_test, input_shape),
            verbose=0,
            return_dict=True,
        )
        print("evaluate metrics:", test_metrics)
    except TypeError:
        test_metrics_list = model.evaluate(
            x_test, supervision_dict_for_fit(y_test, input_shape), verbose=0
        )
        try:
            names = [str(m.name) for m in model.metrics]
            print("evaluate metrics:", dict(zip(names, test_metrics_list)))
        except Exception:  # noqa: BLE001
            print("evaluate raw:", test_metrics_list)

    save_eval_plots_for_split(
        model,
        x_train[:600],
        y_train[:600],
        out_dir,
        "train",
        num_classes=num_classes,
        reject_threshold=float(args.reject_threshold),
        max_points=400,
    )
    save_eval_plots_for_split(
        model,
        x_val,
        y_val,
        out_dir,
        "val",
        num_classes=num_classes,
        reject_threshold=float(args.reject_threshold),
        max_points=400,
    )
    save_eval_plots_for_split(
        model,
        x_test,
        y_test,
        out_dir,
        "test",
        num_classes=num_classes,
        reject_threshold=float(args.reject_threshold),
        max_points=400,
    )
    print(
        "已保存评估图（各 split）：label_compare_*.png、confusion_matrix_*.png、confidence_hist_*.png"
    )

    n_tr = save_attention_mean_by_label_groups(
        model, x_train, y_train, out_dir, "train", group_by="true", num_classes=num_classes
    )
    n_va = save_attention_mean_by_label_groups(
        model, x_val, y_val, out_dir, "val", group_by="true", num_classes=num_classes
    )
    n_te = save_attention_mean_by_label_groups(
        model, x_test, y_test, out_dir, "test", group_by="true", num_classes=num_classes
    )
    print(
        f"已保存按真实标签分组的平均注意力图：train {n_tr} 张，val {n_va} 张，test {n_te} 张（attention_*_by_true_label_*.png）"
    )

    out1 = model.predict(x_test[:1], verbose=0)
    pred_probs = np.asarray(out1["prediction"])
    pred_labels, pmax = predict_with_unknown_reject(
        pred_probs, confidence_threshold=args.reject_threshold
    )
    print("预测类别（-1 为未知）：", int(pred_labels[0]), "p_max=", float(pmax[0]))

    model_structure_png = os.path.abspath(os.path.join(out_dir, "model_structure.png"))
    try:
        keras.utils.plot_model(
            model,
            to_file=model_structure_png,
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",
            dpi=300,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            "未生成 model_structure.png：`keras.utils.plot_model` 需要本机已安装 Graphviz，"
            "且其 `bin` 目录在系统 PATH 中（Windows 上 `dot -V` 能运行）。"
            f"当前错误：{exc}"
        )
        print(
            "处理办法：从 https://graphviz.org/download/ 安装 Graphviz；"
            "或 Conda：`conda install -c conda-forge graphviz`；"
            "并确保已 `pip install pydot`（见 requirements）。"
        )
    else:
        # plot_model 在部分环境下调用 dot 失败时仍可能不抛异常，必须校验文件
        if os.path.isfile(model_structure_png) and os.path.getsize(model_structure_png) > 0:
            print(f"已保存模型结构图：{model_structure_png}")
        else:
            print(
                "未生成有效的 model_structure.png：`plot_model` 已返回，但目标路径下没有非空 PNG。"
                f" 预期文件：{model_structure_png}"
            )
            print(f"系统 PATH 中是否找到 dot：{shutil.which('dot') or '（未找到，请安装 Graphviz 并配置 PATH）'}")

    epochs_ran = len(history.history["loss"])
    ep = np.arange(1, epochs_ran + 1, dtype=np.int32)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(ep, history.history["loss"], label="train loss")
    plt.plot(ep, history.history["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.title("Loss vs epoch")

    plt.subplot(1, 2, 2)
    train_acc_key = "prediction_accuracy" if "prediction_accuracy" in history.history else "accuracy"
    val_acc_key = "val_prediction_accuracy" if "val_prediction_accuracy" in history.history else "val_accuracy"
    plt.plot(ep, history.history.get(train_acc_key, []), label="train acc")
    plt.plot(ep, history.history.get(val_acc_key, []), label="val acc")
    plt.xlabel("epoch")
    plt.legend()
    plt.title("Accuracy vs epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "train_curve.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
