"""
تحليل تجربة الـ 60 إطاراً — مُثبَّت مسبقاً (البند 9 في ANALYSIS_PLAN.md). لا يُعدَّل بعد رؤية النتائج.

H1 (أساسي، TVSum، τ Otani): Δ60 = τ(proposed) − τ(surgery_only) عند 60 إطاراً > 0
    يتحقق إذا كان الحد الأدنى لفترة الثقة 95% (bootstrap، 10,000 إعادة، على مستوى الفيديو) أعلى من الصفر.
H2 (ثانوي): Δ60 − Δ30 > 0 (فرق الفروق على مستوى الفيديو).
وحدة التحليل: الفيديو، بعد أخذ متوسط الـ seeds الثلاثة.

يُشغَّل تلقائياً من run_frames60.py، أو يدوياً: زر الفأرة الأيمن ← Run.
"""
import os
import glob

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
RES = {30: os.path.join(HERE, 'results'), 60: os.path.join(HERE, 'results_f60')}
BACKBONE = 'mobilevitv2_050'
SEEDS = [42, 43, 44]
METRICS = ['tau_ann', 'rho_ann', 'f1_strict']
N_BOOT = 10000


def bootstrap_ci(diff, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon_p(d):
    d = np.asarray(d)
    return float('nan') if len(d) < 5 or np.allclose(d, 0) else float(wilcoxon(d).pvalue)


def load(frames, ds, cfg):
    files = [os.path.join(RES[frames], f"per_video_{ds}_{BACKBONE}_{cfg}_seed{s}.csv") for s in SEEDS]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        return None, 0
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    complete = df.groupby('seed').fold.nunique()
    n_seeds = int((complete == 5).sum())
    return df, n_seeds


def per_video(df, metric):
    return df.groupby('video')[metric].mean()


def cost_row(frames, cfg, ds):
    row = {}
    ms = os.path.join(RES[frames], 'model_stats.csv')
    if os.path.exists(ms):
        s = pd.read_csv(ms)
        s = s[(s.backbone == BACKBONE) & (s.config == cfg)]
        if len(s):
            row['params'] = int(s.params.iloc[-1])
            row['gflops_per_video'] = float(s.gflops_per_video.iloc[-1])
    b = os.path.join(RES[frames], f"baselines_{ds}.csv")
    if os.path.exists(b) and 'decode_ms' in pd.read_csv(b).columns:
        row['decode_ms_median'] = float(pd.read_csv(b).decode_ms.median())
    return row


def analyze(ds):
    data, seeds = {}, {}
    for fr in (30, 60):
        for cfg in ('proposed', 'surgery_only'):
            data[(fr, cfg)], seeds[(fr, cfg)] = load(fr, ds, cfg)
    if any(v is None for v in data.values()):
        missing = [f"{fr}f/{cfg}" for (fr, cfg), v in data.items() if v is None]
        print(f"\n### {ds.upper()}: لا توجد نتائج بعد لـ {', '.join(missing)}\n")
        return None
    final = all(n == len(SEEDS) for n in seeds.values())
    videos = sorted(set.intersection(*[set(v.video) for v in data.values()]))

    print(f"\n## {ds.upper()} — {'نهائي (3 seeds)' if final else 'مبدئي: لم تكتمل الـ seeds الثلاثة'}"
          f" | {len(videos)} فيديو\n")
    print("| المقياس | S30 | P30 | S60 | P60 | Δ30 = P30−S30 | Δ60 = P60−S60 [95% CI] | p (Δ60) "
          "| Δ60−Δ30 [95% CI] |")
    print("|---|---|---|---|---|---|---|---|---|")
    rows = []
    for m in METRICS:
        v = {k: per_video(d, m).loc[videos] for k, d in data.items()}
        d30 = (v[(30, 'proposed')] - v[(30, 'surgery_only')]).to_numpy()
        d60 = (v[(60, 'proposed')] - v[(60, 'surgery_only')]).to_numpy()
        did = d60 - d30
        ci60, cidid = bootstrap_ci(d60), bootstrap_ci(did)
        k = 100 if m.startswith('f1') else 1
        row = dict(dataset=ds, metric=m, n_videos=len(videos), final=final,
                   S30=v[(30, 'surgery_only')].mean(), P30=v[(30, 'proposed')].mean(),
                   S60=v[(60, 'surgery_only')].mean(), P60=v[(60, 'proposed')].mean(),
                   delta30=d30.mean(), delta60=d60.mean(), delta60_ci_low=ci60[0], delta60_ci_high=ci60[1],
                   delta60_wilcoxon_p=wilcoxon_p(d60), did=did.mean(), did_ci_low=cidid[0], did_ci_high=cidid[1],
                   did_wilcoxon_p=wilcoxon_p(did))
        # وصفي: أثر زيادة الإطارات على كل إعداد
        for cfg, tag in (('proposed', 'P'), ('surgery_only', 'S')):
            ch = (v[(60, cfg)] - v[(30, cfg)]).to_numpy()
            row[f'{tag}60_minus_{tag}30'] = ch.mean()
            row[f'{tag}60_minus_{tag}30_ci_low'], row[f'{tag}60_minus_{tag}30_ci_high'] = bootstrap_ci(ch)
        rows.append(row)
        f = (lambda x: f"{x * k:.2f}") if k == 100 else (lambda x: f"{x:.4f}")
        print(f"| {m} | {f(row['S30'])} | {f(row['P30'])} | {f(row['S60'])} | {f(row['P60'])} | "
              f"{row['delta30'] * k:+.4f} | {row['delta60'] * k:+.4f} [{ci60[0] * k:+.4f}, {ci60[1] * k:+.4f}] | "
              f"{row['delta60_wilcoxon_p']:.3f} | {row['did'] * k:+.4f} [{cidid[0] * k:+.4f}, {cidid[1] * k:+.4f}] |")

    print("\n**أثر مضاعفة الإطارات على كل إعداد (τ):**\n")
    t = next(r for r in rows if r['metric'] == 'tau_ann')
    for tag, name in (('P', 'proposed'), ('S', 'surgery_only')):
        print(f"- {name}: {t[f'{tag}60_minus_{tag}30']:+.4f} "
              f"[{t[f'{tag}60_minus_{tag}30_ci_low']:+.4f}, {t[f'{tag}60_minus_{tag}30_ci_high']:+.4f}]")

    print("\n**الكلفة:**\n")
    for fr in (30, 60):
        for cfg in ('proposed', 'surgery_only'):
            c = cost_row(fr, cfg, ds)
            lat = data[(fr, cfg)].groupby('video').latency_ms.median().median()
            print(f"- {fr} إطاراً / {cfg}: GFLOPs لكل فيديو = {c.get('gflops_per_video', float('nan')):.2f} | "
                  f"زمن GPU (وسيط) = {lat:.1f} ms | فك الترميز (وسيط) = {c.get('decode_ms_median', float('nan')):.0f} ms")
    return rows


def main():
    all_rows = []
    print("# تجربة الـ 60 إطاراً — التحليل المُثبَّت مسبقاً (البند 9)\n")
    print("S = surgery_only، P = proposed. الفرق = على مستوى الفيديو بعد متوسط الـ seeds. "
          "فترات الثقة: bootstrap بـ 10,000 إعادة.")
    for ds in ('tvsum', 'summe'):
        r = analyze(ds)
        if r:
            all_rows += r
    if not all_rows:
        return
    pd.DataFrame(all_rows).to_csv(os.path.join(RES[60], 'frames60_analysis.csv'), index=False)

    tv = [r for r in all_rows if r['dataset'] == 'tvsum' and r['metric'] == 'tau_ann']
    print("\n## الحكم (TVSum، τ — قاعدة البند 9)\n")
    if not tv:
        print("لا توجد نتائج TVSum بعد.")
        return
    t = tv[0]
    h1 = t['delta60_ci_low'] > 0
    h2 = t['did_ci_low'] > 0
    status = '' if t['final'] else ' (مبدئي — لم تكتمل الـ seeds الثلاثة؛ الحكم النهائي بعد اكتمالها)'
    print(f"- **H1** (Δ60 > 0): {'تحقق' if h1 else 'لم يتحقق'} — Δ60 = {t['delta60']:+.4f}، "
          f"فترة الثقة [{t['delta60_ci_low']:+.4f}، {t['delta60_ci_high']:+.4f}]{status}")
    print(f"- **H2** (Δ60 − Δ30 > 0): {'تحقق' if h2 else 'لم يتحقق'} — {t['did']:+.4f}، "
          f"فترة الثقة [{t['did_ci_low']:+.4f}، {t['did_ci_high']:+.4f}]")
    if h1:
        print("\n**التفسير المُثبَّت:** الرأس الزمني مبرر عند 60 إطاراً؛ فائدته تظهر مع الإطارات الأكثف "
              "(تُذكر مع كلفتها).")
    else:
        print("\n**التفسير المُثبَّت:** الرأس الزمني لم يُظهر فائدة قابلة للقياس بعد الجراحة، "
              "لا عند 30 ولا عند 60 إطاراً.")


if __name__ == '__main__':
    main()
