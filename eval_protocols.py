"""
بروتوكولات التقييم (بلا PyTorch حتى يمكن اختبارها منفصلة).

strict   : قيمة اللقطة في Knapsack = مجموع الدرجات (بروتوكول الورقة الحالي، مطابق حرفياً
           لـ calculate_shot_f1_fixed_multi_annotator).
standard : قيمة اللقطة = متوسط الدرجات (Zhang et al. 2016، ومعه تحيّز اللقطات القصيرة
           الذي وصفه Otani et al. 2019).
"""
import numpy as np
from scipy.stats import kendalltau, spearmanr

from evaluation_utils_with_fixed import knapsack_summary, _f1_from_binaries

PROTOCOLS = ('strict', 'standard')


def minmax(x):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8) if x.max() > x.min() else x


def summary(scores, shots, capacity, protocol):
    scores = minmax(scores)
    agg = np.sum if protocol == 'strict' else np.mean
    values = np.array([agg(scores[s:e]) for (s, e) in shots], dtype=np.float32)
    weights = np.array([e - s for (s, e) in shots], dtype=int)
    selected = knapsack_summary(values, weights, capacity)
    out = np.zeros(shots[-1][1], dtype=np.float32)
    for sel, (s, e) in zip(selected, shots):
        if sel:
            out[s:e] = 1.0
    return out


def reduce(values, how):
    return float(np.max(values)) if how == 'max' else float(np.mean(values))


def f1_vs_refs(pred_bin, refs, how):
    return reduce([_f1_from_binaries(pred_bin, r) for r in refs], how)


def references(annotators, shots, capacity):
    """ملخص كل مُقيِّم (15%) لكل بروتوكول — يُحسب مرة واحدة لكل فيديو."""
    return {p: [summary(a, shots, capacity, p) for a in annotators] for p in PROTOCOLS}


def random_f1(refs, shots, capacity, n_frames, how, rng, n_draws=20):
    return {p: float(np.mean([f1_vs_refs(summary(rng.random(n_frames), shots, capacity, p), refs[p], how)
                              for _ in range(n_draws)])) for p in PROTOCOLS}


def human_f1(refs, how):
    """السقف البشري: ملخص كل مُقيِّم مقابل الباقين (leave-one-out)."""
    out = {}
    for p in PROTOCOLS:
        r = refs[p]
        out[p] = float(np.mean([f1_vs_refs(r[i], r[:i] + r[i + 1:], how) for i in range(len(r))]))
    return out


def rank_corr(pred, target):
    tau = kendalltau(pred, target)[0]
    rho = spearmanr(pred, target)[0]
    return float(tau), float(rho)


def rank_corr_annotators(pred, annotators, per_annotator):
    """
    per_annotator=True  (TVSum): متوسط τ/ρ مقابل كل مُقيِّم على حدة — طريقة Otani et al.
    per_annotator=False (SumMe): مقابل متوسط المُقيِّمين (ملخصات SumMe ثنائية).
    """
    if not per_annotator:
        return rank_corr(pred, np.mean(annotators, axis=0))
    taus, rhos = zip(*[rank_corr(pred, a) for a in annotators])
    return float(np.nanmean(taus)), float(np.nanmean(rhos))


def human_rank_corr(annotators):
    """τ/ρ البشري: كل مُقيِّم مقابل كل مُقيِّم آخر (TVSum فقط)."""
    taus, rhos = [], []
    for i in range(len(annotators)):
        for j in range(i + 1, len(annotators)):  # τ و ρ متماثلان، فيكفي كل زوج مرة واحدة
            t, r = rank_corr(annotators[i], annotators[j])
            taus.append(t)
            rhos.append(r)
    return float(np.nanmean(taus)), float(np.nanmean(rhos))
