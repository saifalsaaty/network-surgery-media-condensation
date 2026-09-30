"""
Figure 4 of the paper: Kendall's tau on TVSum (mean and standard deviation across seeds) against the CPU latency
per 30-frame video (four threads, FP32) for all 16 configurations.

Input : results/summary_tvsum.csv, results/speed_benchmark.csv, results/baselines_tvsum.csv
Output: results/fig_accuracy_vs_latency.png
Run   : python make_figure4.py
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
S = pd.read_csv(os.path.join(R, 'summary_tvsum.csv'))
SP = pd.read_csv(os.path.join(R, 'speed_benchmark.csv'))
HUMAN_TAU = pd.read_csv(os.path.join(R, 'baselines_tvsum.csv')).human_tau_ann.mean()
# drawing order: the three lightweight backbones (pure, surgery only, full + head, proposed), then the references
ORDER = [f'{b}/{c}' for b in ['mobilevitv2_050', 'mobilevit_xxs', 'mobilenetv3_small_100']
         for c in ['pure', 'surgery_only', 'full_temporal', 'proposed']] + \
        [f'{b}/full_temporal' for b in ['googlenet', 'resnet50', 'mobilenetv3_large_100', 'vit_base_patch16_224']]
T0 = {f'{r.backbone}/{r.config}': {'tau': r.tau_ann, 'tau_sd': r.tau_ann_std} for r in S.itertuples()}
T = {k: T0[k] for k in ORDER}
E = {f'{r.backbone}/{r.config}': {'cpu_ms': r.cpu_fp32_ms} for r in SP.itertuples()}
FAM = {'mobilevitv2_050': ('MobileViT-v2-050', '#2a78d6'), 'mobilevit_xxs': ('MobileViT-XXS', '#eb6834'),
       'mobilenetv3_small_100': ('MobileNetV3-Small', '#1baf7a')}
REFN = {'googlenet': 'GoogLeNet', 'resnet50': 'ResNet-50', 'mobilenetv3_large_100': 'MobileNetV3-Large',
        'vit_base_patch16_224': 'ViT-B/16'}
MK = {'pure': 'o', 'surgery_only': 's', 'full_temporal': '^', 'proposed': 'D'}
CFGN = {'pure': 'Pure', 'surgery_only': 'Surgery only', 'full_temporal': 'Full + temporal head', 'proposed': 'Surgery + temporal head'}
INK, MUTED, GRID, REF = '#0b0b0b', '#52514e', '#e6e5e1', '#52514e'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED})
fig, ax = plt.subplots(figsize=(7.2, 4.9))
ax.axhline(HUMAN_TAU, color=MUTED, lw=1, ls='--', zorder=1)
ax.text(8000, HUMAN_TAU + 0.0015, f'Inter-annotator agreement (τ = {HUMAN_TAU:.3f})', ha='right', va='bottom', fontsize=8, color=MUTED)
for k, v in T.items():
    bb, cfg = k.split('/')
    col = FAM[bb][1] if bb in FAM else REF
    x, y, e = E[k]['cpu_ms'], v['tau'], v['tau_sd']
    ax.errorbar(x, y, yerr=e, fmt='none', ecolor=col, elinewidth=1, capsize=0, alpha=0.6, zorder=2)
    ax.plot(x, y, MK[cfg], ms=8 if cfg != 'proposed' else 9, color=col, mec='white', mew=1.2, zorder=3)
lab = {'mobilevitv2_050/proposed': ('Proposed (0.66 M)', (1150, 0.1235), 'left'),
       'mobilenetv3_small_100/pure': ('MobileNetV3-Small, pure (1.59 M)', (126, 0.1385), 'left'),
       'mobilenetv3_large_100/full_temporal': ('MobileNetV3-Large (4.41 M)', (545, 0.1475), 'right'),
       'googlenet/full_temporal': ('GoogLeNet (5.79 M)', (1900, 0.1575), 'right'),
       'resnet50/full_temporal': ('ResNet-50 (23.8 M)', (2650, 0.1445), 'left'),
       'vit_base_patch16_224/full_temporal': ('ViT-B/16 (86.0 M)', (5900, 0.1665), 'right')}
for k, (t, xy, ha) in lab.items():
    p = (E[k]['cpu_ms'], T[k]['tau'])
    far = k in ('mobilevitv2_050/proposed', 'mobilenetv3_small_100/pure')
    ax.annotate(t, p, xytext=xy, textcoords='data', fontsize=8, color=INK, ha=ha, va='center',
                fontweight='bold' if 'Proposed' in t else 'normal',
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8, shrinkA=2, shrinkB=5) if far else None)
ax.set_xscale('log')
ax.set_xlim(120, 9000)
ax.set_ylim(0.09, 0.19)
ax.set_xticks([150, 300, 600, 1000, 2000, 5000]); ax.set_xticklabels(['150', '300', '600', '1,000', '2,000', '5,000'])
ax.minorticks_off()
ax.set_xlabel('CPU latency per 30-frame video (ms, four threads, FP32, log scale)')
ax.set_ylabel("Kendall's τ on TVSum")
ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
from matplotlib.lines import Line2D
h1 = [Line2D([], [], ls='none', marker='o', ms=7, color=c, mec='white', label=n) for n, c in FAM.values()] + \
     [Line2D([], [], ls='none', marker='^', ms=7, color=REF, mec='white', label='Reference backbones')]
h2 = [Line2D([], [], ls='none', marker=m, ms=7, color=MUTED, mfc='white', label=CFGN[c]) for c, m in MK.items()]
fig.tight_layout(rect=(0, 0.17, 1, 1))
fig.legend(handles=h1, loc='lower left', bbox_to_anchor=(0.08, 0.0), fontsize=7.5, frameon=False, title='Backbone (colour)', title_fontsize=8, ncol=2)
fig.legend(handles=h2, loc='lower right', bbox_to_anchor=(0.98, 0.0), fontsize=7.5, frameon=False, title='Configuration (shape)', title_fontsize=8, ncol=2)
fig.savefig(os.path.join(R, 'fig_accuracy_vs_latency.png'), dpi=300, bbox_inches='tight')
print('[OK]', os.path.join(R, 'fig_accuracy_vs_latency.png'))
