"""
تحليل نتائج F1 بالبروتوكول الرسمي (البند 11). تحليل وصفي استكشافي، لا يُتخذ عليه أي قرار.
يقرأ: results_official/official_f1_per_video.csv و official_baselines.csv و results/summary_<ds>.csv
في PyCharm: زر الفأرة الأيمن ← Run. المخرج: results_official/official_f1_analysis.md (بلا GPU، ثوانٍ)

1) كل إعداد مقابل العشوائي الرسمي: فرق مزدوج لكل فيديو (متوسط الـ seeds)، bootstrap 95% CI، Wilcoxon
2) مقارنات النموذج المقترح (v2-050 proposed) مع البقية بالطريقة نفسها
3) هل يتفق F1 الرسمي مع τ في ترتيب الإعدادات الـ 16؟ (Spearman عبر الإعدادات)
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results_official')
PRIMARY = ('mobilevitv2_050', 'proposed')
N_BOOT, BOOT_SEED = 10000, 0


def boot_ci(d):
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    m = d[idx].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def paired(a, b):
    d = (a - b) * 100
    lo, hi = boot_ci(d)
    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return d.mean(), lo, hi, p


def main():
    df = pd.read_csv(os.path.join(OUT, 'official_f1_per_video.csv'))
    base = pd.read_csv(os.path.join(OUT, 'official_baselines.csv'))
    lines = ["# تحليل F1 الرسمي (البند 11؛ وصفي استكشافي، لا يُتخذ عليه قرار)\n",
             "الفروق بالنقاط المئوية، مزدوجة لكل فيديو بعد أخذ متوسط الـ seeds الثلاثة؛ "
             f"CI = bootstrap 95% ({N_BOOT} عيّنة، seed {BOOT_SEED})؛ p = Wilcoxon.\n"]
    for ds in ['summe', 'tvsum']:
        d = df[df.dataset == ds]
        pv = d.groupby(['backbone', 'config', 'video']).f1_official.mean().unstack('video')
        b = base[base.dataset == ds].set_index('video')
        vids = sorted(set(pv.columns) & set(b.index))
        pv, b = pv[vids], b.loc[vids]
        rnd, hum = b.random_f1.values, b.human_f1.values
        tau = pd.read_csv(os.path.join(HERE, 'results', f'summary_{ds}.csv')).set_index(['backbone', 'config'])

        lines += [f"\n## {ds.upper()} ({len(vids)} فيديو)\n",
                  f"العشوائي الرسمي = {rnd.mean() * 100:.2f}، البشري (leave-one-out) = {hum.mean() * 100:.2f}\n",
                  "| Backbone | Config | F1 (%) | Δ عن العشوائي [95% CI] | p | Kendall τ (بروتوكولنا) |",
                  "|---|---|---|---|---|---|"]
        rows = []
        for (bb, cfg), r in pv.iterrows():
            m, lo, hi, p = paired(r.values, rnd)
            t = tau.loc[(bb, cfg), 'tau_ann'] if (bb, cfg) in tau.index else np.nan
            rows.append((bb, cfg, r.mean() * 100, t))
            lines.append(f"| {bb} | {cfg} | {r.mean() * 100:.2f} | {m:+.2f} [{lo:+.2f}, {hi:+.2f}] | {p:.3f} | {t:.3f} |")

        prim = pv.loc[PRIMARY].values
        lines += [f"\n### {PRIMARY[0]} {PRIMARY[1]} مقابل البقية\n",
                  "| مقابل | Δ F1 [95% CI] | p |", "|---|---|---|"]
        for (bb, cfg), r in pv.iterrows():
            if (bb, cfg) == PRIMARY:
                continue
            m, lo, hi, p = paired(prim, r.values)
            lines.append(f"| {bb} {cfg} | {m:+.2f} [{lo:+.2f}, {hi:+.2f}] | {p:.3f} |")

        s = pd.DataFrame(rows, columns=['bb', 'cfg', 'f1', 'tau']).dropna()
        rs = spearmanr(s.f1, s.tau)
        spread = s.f1.max() - s.f1.min()
        lines += [f"\n**اتفاق F1 الرسمي مع τ في ترتيب الإعدادات ({len(s)}):** Spearman = {rs.statistic:.2f} (p = {rs.pvalue:.3f})",
                  f"\n**مدى F1 بين كل الإعدادات:** {s.f1.min():.2f} – {s.f1.max():.2f} ({spread:.2f} نقطة)، "
                  f"مقابل فرق البشري عن العشوائي {(hum.mean() - rnd.mean()) * 100:+.2f} نقطة.\n"]

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT, 'official_f1_analysis.md'), 'w', encoding='utf-8') as f:
        f.write(text + "\n")
    print(f"\n[✓] انتهى: {os.path.join(OUT, 'official_f1_analysis.md')}")


if __name__ == '__main__':
    main()
