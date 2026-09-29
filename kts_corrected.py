import numpy as np


def _compute_kernel(features, kernel='linear'):
    """يبني مصفوفة النواة (Gram matrix) بين كل زوج من الإطارات."""
    X = features.astype(np.float64)
    if kernel == 'linear':
        K = X @ X.T
    elif kernel == 'rbf':
        sq = np.sum(X ** 2, axis=1)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
        d2 = np.maximum(d2, 0.0)
        gamma = 1.0 / (X.shape[1] + 1e-8)
        K = np.exp(-gamma * d2)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    return K


def kts_segment(features, ncp=None, kernel='linear'):
    """
    Kernel Temporal Segmentation (Potapov et al., ECCV 2014) — نسخة مبسطة
    قائمة على البرمجة الديناميكية لتقليل التباين الداخلي لكل لقطة (segment)
    في فضاء النواة (kernel-induced feature space).

    features : (T, D) — واصف محتوى لكل إطار (لا علاقة له بدرجات الأهمية،
               حتى تكون اللقطات محايدة تماماً تجاه أي انحياز في التقييم).
    ncp      : عدد نقاط التقسيم (Change Points). افتراضياً T//3 — أدق بكثير
               من sqrt(T)، والتي تُنتج لقطات قليلة جداً وخشنة عند T صغير
               (مثلاً T=30 → 5 لقطات فقط)، وهو ما يجعل أي مقارنة لاحقة
               "لعبة اختيار صندوق واحد من أصل قلّة"، ويُضخِّم F1 بشكل
               مصطنع خصوصاً مع بروتوكول max عبر عدة مُقيِّمين.

    يُعيد قائمة حدود اللقطات [(start, end), ...] تغطي [0, T) بالكامل،
    ونفس الحدود تُستخدم لكل من الملخّص المتوقَّع والمرجعي البشري لضمان مقارنة عادلة.
    """
    T = features.shape[0]
    if T <= 1:
        return [(0, max(T, 1))]

    if ncp is None:
        ncp = max(3, T // 3)  # 🔒 مُصحَّح — لا تُرجِعها لـ sqrt(T)
    ncp = min(ncp, T - 1)
    print(f"[KTS-DEBUG] T={T}, ncp={ncp} (expect ~{T // 3} if fixed, ~{int(round(T ** 0.5))} if old)")
    K = _compute_kernel(features, kernel)
    diag = np.diag(K)
    cumdiag = np.concatenate([[0.0], np.cumsum(diag)])

    P = np.zeros((T + 1, T + 1))
    P[1:, 1:] = np.cumsum(np.cumsum(K, axis=0), axis=1)

    def seg_sum(a, b):
        return P[b, b] - P[a, b] - P[b, a] + P[a, a]

    def cost(a, b):
        n = b - a
        if n <= 0:
            return 0.0
        return (cumdiag[b] - cumdiag[a]) - seg_sum(a, b) / n

    m = ncp + 1
    INF = float('inf')
    dp = np.full((m + 1, T + 1), INF)
    split = np.zeros((m + 1, T + 1), dtype=int)
    dp[0, 0] = 0.0

    for k in range(1, m + 1):
        for t in range(k, T + 1):
            best, best_j = INF, k - 1
            for j in range(k - 1, t):
                c = dp[k - 1, j] + cost(j, t)
                if c < best:
                    best, best_j = c, j
            dp[k, t] = best
            split[k, t] = best_j

    cps = []
    t, k = T, m
    while k > 0:
        j = split[k, t]
        if k > 1:
            cps.append(j)
        t, k = j, k - 1

    cps = sorted(set(cps))
    boundaries = [0] + cps + [T]
    shots = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    shots = [s for s in shots if s[1] > s[0]]
    return shots


def frames_to_pooled_features(video_tensor, grid=4):
    """
    واصف محتوى خفيف الوزن لكل إطار، مبني على تجميع مكاني بسيط (Average Pooling).
    """
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        pooled = F.adaptive_avg_pool2d(video_tensor, (grid, grid))
        T = pooled.shape[0]
        feats = pooled.reshape(T, -1).detach().cpu().numpy()
    return feats
