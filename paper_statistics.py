"""
Statistics of the paper that summarize_results.py does not write to a file:

  - Table 3: paired effects of network surgery and of the temporal head (2 x 2 design) on Kendall's tau,
    for each lightweight backbone, with bootstrap 95% confidence intervals and Wilcoxon p-values.
  - Table 1 and Table 7 (last column) and the text: every configuration against the proposed model
    (per-video paired difference in tau, bootstrap 95% CI, Wilcoxon p, TOST p with the +-0.02 margin).
    Comparisons that are not in the pre-specified list (summarize_results.EQ_COMPARE) are marked as
    exploratory; the paper reports them as exploratory and bases no decision on them.

The tests are the ones of summarize_results.py: per-video values averaged over the three seeds,
bootstrap over videos (10,000 resamples, seed 0), two-sided Wilcoxon signed-rank test, TOST.

Input : results/per_video_*.csv
Output: results/paired_effects.csv, results/comparisons_vs_proposed.csv, results/paper_statistics.md
Run   : python paper_statistics.py
"""
import glob
import os

import pandas as pd

import common as C
from summarize_results import EQ_COMPARE, REFERENCE, bootstrap_ci, paired_p, tost_p

METRIC = 'tau_ann'
MARGIN = 0.02
LIGHT = ['mobilevitv2_050', 'mobilevit_xxs', 'mobilenetv3_small_100']
EFFECTS = [  # (name in Table 3, configuration, configuration it is compared with)
    ('surgery without head (surgery only - pure)', 'surgery_only', 'pure'),
    ('head without surgery (full + head - pure)', 'full_temporal', 'pure'),
    ('head after surgery (surgery + head - surgery only)', 'proposed', 'surgery_only'),
    ('surgery with head (surgery + head - full + head)', 'proposed', 'full_temporal'),
]


def per_video(df, backbone, config):
    g = df[(df.backbone == backbone) & (df.config == config)]
    return g.groupby('video')[METRIC].mean()  # mean over the seeds for each video


def paired(a, b):
    common = a.index.intersection(b.index)
    d = (a.loc[common] - b.loc[common]).to_numpy()
    lo, hi = bootstrap_ci(d)
    return dict(n_videos=len(common), mean_diff=float(d.mean()), ci_low=lo, ci_high=hi,
                wilcoxon_p=paired_p(a.loc[common], b.loc[common]), diff=d)


def fmt_d(x):
    x = round(x, 3) + 0.0  # no "-0.000"
    return f'{x:+.3f}' if x else '0.000'


def fmt_p(p):
    return '<0.001' if p < 0.001 else f'{p:.3f}'


def main():
    files = glob.glob(os.path.join(C.RESULTS_DIR, 'per_video_*.csv'))
    df_all = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    effects, comparisons = [], []
    for ds in ['tvsum', 'summe']:
        df = df_all[df_all.dataset == ds]
        for bb in LIGHT:
            for name, a, b in EFFECTS:
                r = paired(per_video(df, bb, a), per_video(df, bb, b))
                r.pop('diff')
                effects.append(dict(dataset=ds, backbone=bb, effect=name, **r))
        ref = per_video(df, *REFERENCE)
        for bb, cfg in df[['backbone', 'config']].drop_duplicates().itertuples(index=False):
            if (bb, cfg) == REFERENCE:
                continue
            r = paired(per_video(df, bb, cfg), ref)
            d = r.pop('diff')
            comparisons.append(dict(dataset=ds, compared=f'{bb}/{cfg}', pre_specified=(bb, cfg) in EQ_COMPARE,
                                    **r, tost_margin=MARGIN, tost_p=tost_p(d, MARGIN)))

    eff = pd.DataFrame(effects)
    cmp_ = pd.DataFrame(comparisons).sort_values(['dataset', 'pre_specified', 'compared'],
                                                 ascending=[False, False, True])
    eff.to_csv(os.path.join(C.RESULTS_DIR, 'paired_effects.csv'), index=False)
    cmp_.to_csv(os.path.join(C.RESULTS_DIR, 'comparisons_vs_proposed.csv'), index=False)

    lines = []
    for ds in ['tvsum', 'summe']:
        table3 = ' (Table 3)' if ds == 'tvsum' else ''
        lines += [f"### {ds.upper()}: paired effects on Kendall's tau{table3}", '',
                  '| Backbone | Effect | Difference [95% CI] | Wilcoxon p |', '|---|---|---|---|']
        for r in eff[eff.dataset == ds].itertuples():
            lines.append(f'| {r.backbone} | {r.effect} | {fmt_d(r.mean_diff)} [{fmt_d(r.ci_low)}, {fmt_d(r.ci_high)}] '
                         f'| {fmt_p(r.wilcoxon_p)} |')
        lines += ['', f'### {ds.upper()}: each configuration against the proposed model (compared - proposed, tau)', '',
                  '| Compared | Pre-specified | Difference [95% CI] | Wilcoxon p | TOST p (+-0.02) |',
                  '|---|---|---|---|---|']
        for r in cmp_[cmp_.dataset == ds].itertuples():
            lines.append(f'| {r.compared} | {"yes" if r.pre_specified else "no (exploratory)"} | '
                         f'{fmt_d(r.mean_diff)} [{fmt_d(r.ci_low)}, {fmt_d(r.ci_high)}] | {fmt_p(r.wilcoxon_p)} | '
                         f'{fmt_p(r.tost_p)} |')
        lines.append('')
    text = '\n'.join(lines)
    with open(os.path.join(C.RESULTS_DIR, 'paper_statistics.md'), 'w', encoding='utf-8') as f:
        f.write(text)
    print(text)


if __name__ == '__main__':
    main()
