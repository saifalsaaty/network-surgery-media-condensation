"""
يجمع كل نتائج results/per_video_*.csv في جداول جاهزة للورقة:
  - results/summary_<dataset>.csv  : الدقة (mean ± std عبر الـ seeds) + اختبارات Wilcoxon
  - results/efficiency.csv         : المعاملات، الحجم، GFLOPs، الزمن، الذاكرة، وعامل الزمن الحقيقي
ويطبع الجداول بصيغة Markdown.
"""
import os
import glob

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import common as C

METRICS = ['f1_strict', 'f1_standard', 'f1_strict_inverted', 'tau_ann', 'rho_ann', 'tau_meangt30', 'rho_meangt30']


def paired_p(a, b):
    """Wilcoxon ثنائي الطرف على مستوى الفيديو."""
    d = np.asarray(a) - np.asarray(b)
    if len(d) < 5 or np.allclose(d, 0):
        return float('nan')
    return float(wilcoxon(a, b).pvalue)


def fmt(m, s, pct):
    k = 100 if pct else 1
    return f"{m * k:.2f} ± {s * k:.2f}" if pct else f"{m:.4f} ± {s:.4f}"


def summarize_dataset(df, base, dataset):
    rows = []
    for (bb, cfg), g in df.groupby(['backbone', 'config']):
        per_seed = g.groupby('seed')[METRICS].mean()
        fold_std = g.groupby(['seed', 'fold'])['f1_strict'].mean().groupby('seed').std().mean()
        per_video = g.groupby('video')[METRICS].mean()  # متوسط عبر الـ seeds لكل فيديو
        b = base.set_index('video').loc[per_video.index]
        row = dict(backbone=bb, config=cfg, n_seeds=len(per_seed), n_videos=len(per_video))
        for m in METRICS:
            row[m] = per_seed[m].mean()
            row[m + '_std'] = per_seed[m].std(ddof=1) if len(per_seed) > 1 else 0.0
        # المقياس المُطبَّع: 0 = عشوائي، 1 = سقف بشري
        r, h = base.random_f1_strict.mean(), base.human_f1_strict.mean()
        norm = (per_seed['f1_strict'] - r) / (h - r)
        row['norm_strict'] = norm.mean()
        row['norm_strict_std'] = norm.std(ddof=1) if len(norm) > 1 else 0.0
        row['f1_strict_fold_std'] = fold_std
        row['p_vs_random_strict'] = paired_p(per_video.f1_strict, b.random_f1_strict)
        row['p_vs_random_standard'] = paired_p(per_video.f1_standard, b.random_f1_standard)
        pure = df[(df.backbone == bb) & (df.config == 'pure')]
        if cfg != 'pure' and len(pure):
            pv = pure.groupby('video')[METRICS].mean().loc[per_video.index]
            row['p_vs_pure_f1_strict'] = paired_p(per_video.f1_strict, pv.f1_strict)
            row['p_vs_pure_tau_ann'] = paired_p(per_video.tau_ann, pv.tau_ann)
        # مقارنة كل إعداد بالنموذج المقترح الأساسي (v2-050 + Conv1d-Transformer)
        prop = df[(df.backbone == 'mobilevitv2_050') & (df.config == 'proposed')]
        if not (bb == 'mobilevitv2_050' and cfg == 'proposed') and len(prop):
            pp = prop.groupby('video')[METRICS].mean().loc[per_video.index]
            row['p_vs_proposed_f1_strict'] = paired_p(per_video.f1_strict, pp.f1_strict)
            row['p_vs_proposed_tau_ann'] = paired_p(per_video.tau_ann, pp.tau_ann)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(C.RESULTS_DIR, f"summary_{dataset}.csv"), index=False)

    # جدول Markdown
    border = {b: i for i, b in enumerate(C.ALL_BACKBONES)}
    corder = {c: i for i, c in enumerate(C.CONFIGS)}
    out = out.assign(_b=out.backbone.map(border).fillna(99), _c=out.config.map(corder).fillna(99))
    out = out.sort_values(['_b', '_c']).drop(columns=['_b', '_c'])
    get = lambda r, k: r[k] if k in r and pd.notna(r[k]) else float('nan')
    print(f"\n### {dataset.upper()}  (mean ± std عبر الـ seeds؛ F1 بالنسبة المئوية)\n")
    print("| Backbone | Config | F1 strict | F1 standard | Norm. strict | τ (ann) | ρ (ann) "
          "| p vs random | p vs pure (F1) | p vs proposed (τ) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in out.iterrows():
        print(f"| {r.backbone} | {r.config} | {fmt(r.f1_strict, r.f1_strict_std, True)} | "
              f"{fmt(r.f1_standard, r.f1_standard_std, True)} | {r.norm_strict * 100:.1f}% | "
              f"{fmt(r.tau_ann, r.tau_ann_std, False)} | {fmt(r.rho_ann, r.rho_ann_std, False)} | "
              f"{r.p_vs_random_strict:.3f} | {get(r, 'p_vs_pure_f1_strict'):.3f} | "
              f"{get(r, 'p_vs_proposed_tau_ann'):.3f} |")
    hum_tau = f"{base.human_tau_ann.mean():.4f}" if 'human_tau_ann' in base else "—"
    print(f"| — | Random | {base.random_f1_strict.mean() * 100:.2f} | {base.random_f1_standard.mean() * 100:.2f} "
          f"| 0% | 0 | 0 | | | |")
    print(f"| — | Human (LOO) | {base.human_f1_strict.mean() * 100:.2f} | {base.human_f1_standard.mean() * 100:.2f} "
          f"| 100% | {hum_tau} | | | | |")


def efficiency(df_all, bases):
    stats_path = os.path.join(C.RESULTS_DIR, "model_stats.csv")
    stats = pd.read_csv(stats_path) if os.path.exists(stats_path) else pd.DataFrame()
    rows = []
    for (ds, bb, cfg), g in df_all.groupby(['dataset', 'backbone', 'config']):
        base = bases[ds].set_index('video')
        lat = g.groupby('video').latency_ms.median()
        dur_ms = base.loc[lat.index, 'duration_s'] * 1000
        row = dict(dataset=ds, backbone=bb, config=cfg, latency_ms=lat.median(),
                   frames_per_s=C.NUM_FRAMES / (lat.median() / 1000),
                   peak_mem_mb=g.peak_mem_mb.median(),
                   realtime_factor_model=float(np.median(dur_ms / lat)))
        if 'decode_ms' in base:
            total = lat + base.loc[lat.index, 'decode_ms']
            row['decode_ms'] = base.loc[lat.index, 'decode_ms'].median()
            row['realtime_factor_end_to_end'] = float(np.median(dur_ms / total))
        rows.append(row)
    eff = pd.DataFrame(rows)
    if len(stats):
        eff = eff.merge(stats, on=['backbone', 'config'], how='left')
    eff.to_csv(os.path.join(C.RESULTS_DIR, "efficiency.csv"), index=False)
    print("\n### Efficiency (الزمن = وسيط ms لكل فيديو من 30 إطاراً على الـ GPU)\n")
    cols = [c for c in ['dataset', 'backbone', 'config', 'params', 'size_fp16_mb', 'gflops_per_video',
                        'latency_ms', 'peak_mem_mb', 'realtime_factor_model', 'decode_ms',
                        'realtime_factor_end_to_end'] if c in eff]
    try:
        print(eff[cols].to_markdown(index=False, floatfmt='.2f'))
    except ImportError:  # الحزمة tabulate غير مثبتة
        print(eff[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))


# ------------------------------------------------------------
# اختبار التكافؤ: "عدد معاملات أقل بدقة مكافئة"
# ------------------------------------------------------------
# هوامش التكافؤ مُحدَّدة مسبقاً قبل اكتمال النتائج — لا تُغيَّر بعد رؤيتها.
#   τ: 0.02  (≈ 11% من τ البشري على TVSum = 0.179)
#   F1 strict: 1.5 نقطة (≈ 17% من الفجوة بين العشوائي والبشري على TVSum = 8.6 نقطة)
EQ_MARGINS = {'tau_ann': 0.02, 'f1_strict': 0.015}
REFERENCE = ('mobilevitv2_050', 'proposed')
EQ_COMPARE = [('mobilevitv2_050', 'full_temporal'), ('mobilevitv2_050', 'pure'), ('mobilevit_xxs', 'proposed'),
              ('mobilenetv3_small_100', 'proposed'), ('mobilenetv3_large_100', 'full_temporal'),
              ('googlenet', 'full_temporal'), ('resnet50', 'full_temporal'), ('vit_base_patch16_224', 'full_temporal')]


def bootstrap_ci(diff, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def tost_p(diff, margin):
    """Two One-Sided Tests: p < 0.05 يعني أن الفرق ضمن ±margin (تكافؤ)."""
    from scipy.stats import ttest_1samp
    p_low = ttest_1samp(diff, -margin, alternative='greater').pvalue
    p_high = ttest_1samp(diff, margin, alternative='less').pvalue
    return float(max(p_low, p_high))


def equivalence(df, dataset):
    """
    المرجع: النموذج المقترح (v2-050). الفرق = المقارَن − المقترح، لكل فيديو (متوسط عبر الـ seeds).
    الحكم:
      equivalent  : فترة الثقة 95% كلها داخل ±الهامش
      better/worse: فترة الثقة كلها خارج الهامش (المقارَن أفضل/أسوأ بوضوح)
      inconclusive: غير ذلك (البيانات لا تكفي للحكم)
    """
    stats_path = os.path.join(C.RESULTS_DIR, 'model_stats.csv')
    params = pd.read_csv(stats_path).set_index(['backbone', 'config']).params if os.path.exists(stats_path) else {}
    ref = df[(df.backbone == REFERENCE[0]) & (df.config == REFERENCE[1])]
    if not len(ref):
        return
    ref_v = ref.groupby('video')[list(EQ_MARGINS)].mean()
    rows = []
    for bb, cfg in EQ_COMPARE:
        g = df[(df.backbone == bb) & (df.config == cfg)]
        if not len(g):
            continue
        v = g.groupby('video')[list(EQ_MARGINS)].mean()
        common = v.index.intersection(ref_v.index)
        for m, margin in EQ_MARGINS.items():
            diff = (v.loc[common, m] - ref_v.loc[common, m]).to_numpy()
            lo, hi = bootstrap_ci(diff)
            if -margin < lo and hi < margin:
                verdict = 'equivalent'
            elif lo > margin:
                verdict = 'better'
            elif hi < -margin:
                verdict = 'worse'
            else:
                verdict = 'inconclusive'
            p_ref = params.get(REFERENCE, np.nan) if hasattr(params, 'get') else np.nan
            p_cmp = params.get((bb, cfg), np.nan) if hasattr(params, 'get') else np.nan
            rows.append(dict(dataset=dataset, compared=f"{bb}/{cfg}", metric=m, n_videos=len(common),
                             n_seeds=g.seed.nunique(), mean_diff=float(diff.mean()), ci_low=lo, ci_high=hi,
                             margin=margin, tost_p=tost_p(diff, margin), verdict=verdict,
                             params_compared=p_cmp, params_ratio_vs_proposed=p_cmp / p_ref if p_ref else np.nan))
    if not rows:
        return
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(C.RESULTS_DIR, f"equivalence_{dataset}.csv"), index=False)
    print(f"\n### {dataset.upper()} — اختبار التكافؤ مقابل النموذج المقترح v2-050 "
          f"(الفرق = المقارَن − المقترح؛ فترة ثقة 95% bootstrap؛ TOST)\n")
    print("| Compared | Params × proposed | Metric | Δ mean | 95% CI | margin | TOST p | Verdict |")
    print("|---|---|---|---|---|---|---|---|")
    for _, r in out.iterrows():
        k = 100 if r.metric.startswith('f1') else 1
        print(f"| {r.compared} | {r.params_ratio_vs_proposed:.2f}× | {r.metric} | {r.mean_diff * k:+.3f} | "
              f"[{r.ci_low * k:+.3f}, {r.ci_high * k:+.3f}] | ±{r.margin * k:g} | {r.tost_p:.3f} | {r.verdict} |")


# ------------------------------------------------------------
# قاعدة اختيار النموذج الأساسي (مُثبَّتة مسبقاً — البند 8 في ANALYSIS_PLAN.md)
# ------------------------------------------------------------
# الأساسي = أقل المرشحين معاملاتٍ بشرط عدم الدونية في τ على TVSum مقابل v2-050 proposed:
# الحد الأدنى لفترة الثقة 95% (bootstrap، على مستوى الفيديو، متوسط الـ seeds الثلاثة) للفرق > −0.02.
# إن لم يحقق أي مرشح أصغر ذلك، يبقى v2-050 proposed. SumMe لا يدخل في القرار.
SELECTION_CANDIDATES = [('mobilevitv2_050', 'proposed'), ('mobilevit_xxs', 'proposed'),
                        ('mobilenetv3_small_100', 'proposed')]
SELECTION_DATASET, SELECTION_METRIC, NI_MARGIN = 'tvsum', 'tau_ann', 0.02


def select_primary(df):
    stats_path = os.path.join(C.RESULTS_DIR, 'model_stats.csv')
    if not os.path.exists(stats_path):
        return
    params = pd.read_csv(stats_path).drop_duplicates(['backbone', 'config'], keep='last') \
        .set_index(['backbone', 'config']).params
    d = df[df.dataset == SELECTION_DATASET]
    ref = d[(d.backbone == REFERENCE[0]) & (d.config == REFERENCE[1])]
    if not len(ref):
        return
    ref_v = ref.groupby('video')[SELECTION_METRIC].mean()
    rows, complete = [], True
    for bb, cfg in SELECTION_CANDIDATES:
        g = d[(d.backbone == bb) & (d.config == cfg)]
        if not len(g) or (bb, cfg) not in params.index:
            complete = False
            continue
        complete &= g.seed.nunique() == len(C.SEEDS) and ref.seed.nunique() == len(C.SEEDS)
        v = g.groupby('video')[SELECTION_METRIC].mean()
        common = v.index.intersection(ref_v.index)
        diff = (v.loc[common] - ref_v.loc[common]).to_numpy()
        lo, hi = bootstrap_ci(diff) if (bb, cfg) != REFERENCE else (0.0, 0.0)
        rows.append(dict(candidate=f"{bb}/{cfg}", params=int(params[(bb, cfg)]), n_seeds=g.seed.nunique(),
                         tau=float(v.mean()), delta=float(diff.mean()), ci_low=lo, ci_high=hi,
                         non_inferior=(bb, cfg) == REFERENCE or lo > -NI_MARGIN))
    out = pd.DataFrame(rows).sort_values('params')
    out.to_csv(os.path.join(C.RESULTS_DIR, 'primary_selection.csv'), index=False)
    ok = out[out.non_inferior]
    chosen = ok.iloc[0].candidate if len(ok) else f"{REFERENCE[0]}/{REFERENCE[1]}"
    print(f"\n### اختيار النموذج الأساسي (قاعدة مُثبَّتة مسبقاً: عدم الدونية في τ على TVSum، الهامش {NI_MARGIN})\n")
    print("| Candidate | Params | Seeds | τ | Δ vs v2-050 | 95% CI | Non-inferior |")
    print("|---|---|---|---|---|---|---|")
    for _, r in out.iterrows():
        print(f"| {r.candidate} | {r.params:,} | {r.n_seeds} | {r.tau:.4f} | {r.delta:+.4f} | "
              f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}] | {'yes' if r.non_inferior else 'no'} |")
    status = "نهائي" if complete and len(out) == len(SELECTION_CANDIDATES) else "مبدئي — لم تكتمل الـ seeds الثلاثة لكل المرشحين"
    print(f"\n**النموذج الأساسي: {chosen}** ({status})")


def main():
    files = glob.glob(os.path.join(C.RESULTS_DIR, "per_video_*.csv"))
    if not files:
        print("لا توجد نتائج بعد. شغّل evaluate_cv.py أولاً.")
        return
    df_all = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    bases = {}
    for ds in C.DATASETS:
        p = os.path.join(C.RESULTS_DIR, f"baselines_{ds}.csv")
        if os.path.exists(p) and (df_all.dataset == ds).any():
            bases[ds] = pd.read_csv(p)
            summarize_dataset(df_all[df_all.dataset == ds], bases[ds], ds)
            try:  # لا يُسمح لهذا القسم الإضافي أن يوقف التشغيل الطويل أبداً
                equivalence(df_all[df_all.dataset == ds], ds)
            except Exception as ex:
                print(f"[!] تخطّي اختبار التكافؤ لـ {ds}: {ex}")
    try:
        select_primary(df_all)
    except Exception as ex:
        print(f"[!] تخطّي اختيار النموذج الأساسي: {ex}")
    efficiency(df_all[df_all.dataset.isin(bases)], bases)


if __name__ == '__main__':
    main()
