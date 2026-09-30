import numpy as np
import torch


def knapsack_summary(values, weights, capacity):
    n = len(values)
    dp = [[0.0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        v = values[i - 1]
        w = weights[i - 1]
        for c in range(1, capacity + 1):
            if w <= c:
                dp[i][c] = max(dp[i - 1][c], dp[i - 1][c - w] + v)
            else:
                dp[i][c] = dp[i - 1][c]
    c = capacity
    selected = np.zeros(n, dtype=np.float32)
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i - 1][c]:
            selected[i - 1] = 1.0
            c -= weights[i - 1]
    return selected


def _shot_binary_from_scores(scores, shots, capacity):
    values = np.array([scores[s:e].sum() for (s, e) in shots], dtype=np.float32)
    weights = np.array([e - s for (s, e) in shots], dtype=int)
    shot_selected = knapsack_summary(values, weights, capacity)
    total_frames = shots[-1][1]
    frame_binary = np.zeros(total_frames, dtype=np.float32)
    for sel, (s, e) in zip(shot_selected, shots):
        if sel:
            frame_binary[s:e] = 1.0
    return frame_binary


def _f1_from_binaries(pred_binary, true_binary):
    tp = (pred_binary * true_binary).sum()
    fp = (pred_binary * (1 - true_binary)).sum()
    fn = ((1 - pred_binary) * true_binary).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return float(2 * (precision * recall) / (precision + recall + 1e-8))


def calculate_shot_f1_multi_annotator(pred_scores, annotator_scores, shots, capacity_ratio=0.15):
    pred_np = pred_scores.cpu().float().numpy() if isinstance(pred_scores, torch.Tensor) else np.asarray(pred_scores, dtype=np.float32)
    if pred_np.max() > pred_np.min():
        pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)

    total_frames = len(pred_np)
    capacity = max(1, int(capacity_ratio * total_frames))
    pred_binary = _shot_binary_from_scores(pred_np, shots, capacity)

    ann_np = annotator_scores.cpu().float().numpy() if isinstance(annotator_scores, torch.Tensor) else np.asarray(annotator_scores, dtype=np.float32)

    f1_per_annotator = []
    for ann in ann_np:
        if ann.max() > ann.min():
            ann_norm = (ann - ann.min()) / (ann.max() - ann.min() + 1e-8)
        else:
            ann_norm = ann
        true_binary = _shot_binary_from_scores(ann_norm, shots, capacity)
        f1_per_annotator.append(_f1_from_binaries(pred_binary, true_binary))

    return float(np.mean(f1_per_annotator))


def calculate_shot_f1_dynamic_multi_annotator(pred_scores, annotator_binary, shots, reduction='max'):
    pred_np = pred_scores.cpu().float().numpy() if isinstance(pred_scores, torch.Tensor) else np.asarray(pred_scores, dtype=np.float32)
    if pred_np.max() > pred_np.min():
        pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)

    ann_np = annotator_binary.cpu().float().numpy() if isinstance(annotator_binary, torch.Tensor) else np.asarray(annotator_binary, dtype=np.float32)

    f1_per_annotator = []
    for user_row in ann_np:
        capacity = max(1, int(round(user_row.sum())))
        pred_binary = _shot_binary_from_scores(pred_np, shots, capacity)
        true_binary = _shot_binary_from_scores(user_row, shots, capacity)
        f1_per_annotator.append(_f1_from_binaries(pred_binary, true_binary))

    if not f1_per_annotator:
        return 0.0
    if reduction == 'max':
        return float(np.max(f1_per_annotator))
    elif reduction == 'avg':
        return float(np.mean(f1_per_annotator))
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


# ============================================================
# 🔬 جديدة: سعة ثابتة (15%) لكلا الطرفين — تطابق بروتوكول Zhang et al. 2016 /
# DR-DSN / VASNet / CSNet (Table 8)، بعكس السعة الديناميكية (طول ملخّص كل
# مُقيِّم) في بروتوكول Gygli الأصلي المُستخدَم في calculate_shot_f1_dynamic_multi_annotator.
# ============================================================
def calculate_shot_f1_fixed_multi_annotator(pred_scores, annotator_binary, shots,
                                            capacity_ratio=0.15, reduction='max'):
    pred_np = pred_scores.cpu().float().numpy() if isinstance(pred_scores, torch.Tensor) else np.asarray(pred_scores, dtype=np.float32)
    if pred_np.max() > pred_np.min():
        pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)

    total_frames = len(pred_np)
    capacity = max(1, int(capacity_ratio * total_frames))
    pred_binary = _shot_binary_from_scores(pred_np, shots, capacity)

    ann_np = annotator_binary.cpu().float().numpy() if isinstance(annotator_binary, torch.Tensor) else np.asarray(annotator_binary, dtype=np.float32)

    f1_per_annotator = []
    for user_row in ann_np:
        # 🔧 إصلاح: تطبيع كل مُقيِّم قبل بناء قيم Knapsack — ضروري لـTVSum
        # (درجات خام 1-5) لمنع انحياز نحو اللقطات الأطول؛ بلا أثر على SumMe
        # (بيانات ثنائية 0/1 أصلاً، لا يُغيِّرها التطبيع).
        if user_row.max() > user_row.min():
            user_row_norm = (user_row - user_row.min()) / (user_row.max() - user_row.min() + 1e-8)
        else:
            user_row_norm = user_row
        true_binary = _shot_binary_from_scores(user_row_norm, shots, capacity)
        f1_per_annotator.append(_f1_from_binaries(pred_binary, true_binary))

    if not f1_per_annotator:
        return 0.0
    if reduction == 'max':
        return float(np.max(f1_per_annotator))
    elif reduction == 'avg':
        return float(np.mean(f1_per_annotator))
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
