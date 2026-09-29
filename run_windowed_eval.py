"""
تجربة الاستدلال بالنوافذ على الفيديو كاملاً ("30 إطاراً من كل 100") — البند 10 في ANALYSIS_PLAN.md.
استدلال فقط بالنماذج المدرَّبة (mobilevitv2_050 / proposed، آخر epoch، 3 seeds)؛ لا يوجد أي تدريب.

في PyCharm: زر الفأرة الأيمن ← Run 'run_windowed_eval'. المدة المتوقعة: نحو 20–45 دقيقة (أغلبها فك ترميز).
- قابل للاستئناف: يتخطى الفيديوهات التي قُيِّمت مسبقاً.
- النتائج في results_windowed/ ؛ والتحليل المُثبَّت مسبقاً في results_windowed/windowed_analysis.md
- لا يلمس أي نتيجة سابقة.
"""
import os
import sys
import time

os.environ.pop('VS_NUM_FRAMES', None)  # يعمل على cache الـ 30 إطاراً والـ checkpoints الأصلية فقط

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

import common as C
import eval_protocols as E

WINDOW, PER_WINDOW = 100, 30
BACKBONE, CONFIG = 'mobilevitv2_050', 'proposed'
SEEDS = [42, 43, 44]
DATASETS = ['tvsum', 'summe']  # TVSum (الأساسي) أولاً
OUT = os.path.join(C.WORK_DIR, 'results_windowed')
NI_MARGIN, N_BOOT = 0.02, 10000


def capacity():
    return max(1, int(C.CAPACITY_RATIO * C.FINE_FRAMES))


def window_indices(n):
    """نوافذ متتالية غير متداخلة من 100 إطار؛ من كل نافذة حتى 30 إطاراً موزعة بالتساوي."""
    wins = []
    for s in range(0, n, WINDOW):
        e = min(s + WINDOW, n)
        k = min(PER_WINDOW, e - s)
        wins.append(np.unique(np.linspace(s, e - 1, k).round().astype(int)))
    return wins


def score_video(path, n, models):
    """قراءة متسلسلة للفيديو، نافذة بعد نافذة؛ كل نافذة تمر بالنماذج وحدها ثم تُحذف من الذاكرة."""
    import cv2
    wins = window_indices(n)
    cap = cv2.VideoCapture(path)
    pos, last = 0, None
    idx_all, scores = [], {s: [] for s in models}
    t_decode = t_model = 0.0
    for w in wins:
        t0 = time.perf_counter()
        frames = []
        for fi in w:
            while pos < fi:          # تخطّي الإطارات غير المطلوبة بلا فك ترميز كامل
                cap.grab()
                pos += 1
            ok = cap.grab()
            pos += 1
            fr = None
            if ok:
                ok, fr = cap.retrieve()
            if ok and fr is not None:
                fr = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), (C.IMAGE_SIZE, C.IMAGE_SIZE))
                last = fr
            else:
                fr = last if last is not None else np.zeros((C.IMAGE_SIZE, C.IMAGE_SIZE, 3), np.uint8)
            frames.append(fr)
        t_decode += time.perf_counter() - t0

        x = torch.from_numpy(np.stack(frames).transpose(0, 3, 1, 2)).float().div(255.0).unsqueeze(0).to(C.DEVICE)
        t0 = time.perf_counter()
        with torch.no_grad():
            for s, m in models.items():
                scores[s].append(m(x).float().view(-1).cpu().numpy())
        if C.DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        t_model += time.perf_counter() - t0
        idx_all.append(w)
    cap.release()
    idx = np.concatenate(idx_all)
    return idx, {s: np.concatenate(v) for s, v in scores.items()}, len(wins), t_decode, t_model


def evaluate_dataset(ds):
    out_path = os.path.join(OUT, f"per_video_windowed_{ds}.csv")
    done = pd.read_csv(out_path) if os.path.exists(out_path) else pd.DataFrame(columns=['video'])
    done_videos = set(done.video)
    rows = done.to_dict('records')
    cache = C.load_cache(ds)
    folds = C.get_folds(len(cache))
    how = C.HPARAMS[ds]['reduction']
    for k, (_, test_idx) in enumerate(folds, start=1):
        todo = [i for i in test_idx if cache[i]['name'] not in done_videos]
        if not todo:
            continue
        models = {}
        for s in C.SEEDS:
            ckpt = os.path.join(C.run_dir(ds, BACKBONE, CONFIG, s), f"fold_{k}_last.pth")
            m = C.build_model(BACKBONE, CONFIG, pretrained=False).to(C.DEVICE)
            m.load_state_dict(torch.load(ckpt, map_location=C.DEVICE, weights_only=True), strict=True)
            models[s] = m.eval()
        for i in todo:
            item = cache[i]
            n = int(item['n_frames'])
            idx, sc, n_win, t_dec, t_mod = score_video(C.video_file(ds, item['name']), n, models)
            fine = np.linspace(0, n - 1, C.FINE_FRAMES, dtype=int)
            ann = item['annotators'].numpy()
            refs = E.references(ann, item['shots'], capacity())
            for s in C.SEEDS:
                p150 = np.interp(fine, idx, sc[s])
                f1 = {p: E.f1_vs_refs(E.summary(p150, item['shots'], capacity(), p), refs[p], how)
                      for p in E.PROTOCOLS}
                tau, rho = E.rank_corr_annotators(p150, ann, per_annotator=(ds == 'tvsum'))
                rows.append(dict(dataset=ds, seed=s, fold=k, video=item['name'], n_frames=n, n_windows=n_win,
                                 n_sampled=len(idx), f1_strict=f1['strict'], f1_standard=f1['standard'],
                                 tau_ann=tau, rho_ann=rho, decode_s=t_dec, model_s_all_seeds=t_mod))
            pd.DataFrame(rows).to_csv(out_path, index=False)  # حفظ تدريجي
            print(f"[{ds} | fold {k}] {item['name']}: {len(idx)} إطاراً في {n_win} نافذة | "
                  f"τ (متوسط الـ seeds) = {np.mean([r['tau_ann'] for r in rows[-3:]]):.4f} | "
                  f"فك الترميز {t_dec:.1f} s", flush=True)
        del models
        if C.DEVICE.type == 'cuda':
            torch.cuda.empty_cache()


# ------------------------------------------------------------
# التحليل المُثبَّت مسبقاً (البند 10) — لا يُعدَّل بعد رؤية النتائج
# ------------------------------------------------------------
def bootstrap_ci(d, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(N_BOOT, len(d)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def analyze():
    lines = ["# الاستدلال بالنوافذ (30 من كل 100) مقابل 30 إطاراً موزعة — التحليل المُثبَّت مسبقاً (البند 10)\n",
             "Δ = النوافذ − الموزعة، على مستوى الفيديو بعد متوسط الـ seeds الثلاثة؛ فترة ثقة bootstrap بـ 10,000 إعادة.\n"]
    verdict = None
    for ds in DATASETS:
        wp = os.path.join(OUT, f"per_video_windowed_{ds}.csv")
        up = [os.path.join(C.RESULTS_DIR, f"per_video_{ds}_{BACKBONE}_{CONFIG}_seed{s}.csv") for s in SEEDS]
        if not os.path.exists(wp) or not all(os.path.exists(p) for p in up):
            lines.append(f"\n## {ds.upper()}: لا توجد نتائج كاملة بعد.\n")
            continue
        w = pd.read_csv(wp)
        u = pd.concat([pd.read_csv(p) for p in up], ignore_index=True)
        complete = w.groupby('video').seed.nunique().eq(len(SEEDS))
        vids = sorted(set(complete[complete].index) & set(u.video))
        n_total = u.video.nunique()
        lines.append(f"\n## {ds.upper()} — {len(vids)}/{n_total} فيديو"
                     f"{'' if len(vids) == n_total else ' (مبدئي: لم تكتمل كل الفيديوهات)'}\n")
        lines.append("| المقياس | 30 موزعة | النوافذ | Δ [95% CI] | p (Wilcoxon) |")
        lines.append("|---|---|---|---|---|")
        for m in ['tau_ann', 'rho_ann', 'f1_strict', 'f1_standard']:
            a = w[w.video.isin(vids)].groupby('video')[m].mean().loc[vids]
            b = u[u.video.isin(vids)].groupby('video')[m].mean().loc[vids]
            d = (a - b).to_numpy()
            lo, hi = bootstrap_ci(d)
            p = float(wilcoxon(d).pvalue) if len(d) >= 5 and not np.allclose(d, 0) else float('nan')
            k, nd = (100, 2) if m.startswith('f1') else (1, 4)
            lines.append(f"| {m} | {b.mean() * k:.{nd}f} | {a.mean() * k:.{nd}f} | {d.mean() * k:+.{nd}f} "
                         f"[{lo * k:+.{nd}f}, {hi * k:+.{nd}f}] | {p:.3f} |")
            if ds == 'tvsum' and m == 'tau_ann':
                verdict = (d.mean(), lo, hi, len(vids) == n_total)
        ww = w[w.video.isin(vids)].drop_duplicates('video')
        lines.append(f"\n- الإطارات المعالَجة لكل فيديو (وسيط): {ww.n_sampled.median():.0f} مقابل 30، "
                     f"في {ww.n_windows.median():.0f} نافذة")
        lines.append(f"- زمن فك الترميز لكل فيديو (وسيط): {ww.decode_s.median():.1f} s؛ "
                     f"زمن النموذج لكل فيديو (وسيط، لـ seed واحد): {ww.model_s_all_seeds.median() / len(SEEDS) * 1000:.0f} ms")

    lines.append("\n## الحكم (TVSum، τ — قاعدة البند 10)\n")
    if verdict is None:
        lines.append("لا توجد نتائج TVSum كاملة بعد.")
    else:
        dm, lo, hi, final = verdict
        tag = '' if final else ' (مبدئي)'
        if lo > 0:
            v = "تفوّق دال: الاستدلال بالنوافذ أفضل من 30 إطاراً موزعة."
        elif lo > -NI_MARGIN:
            v = "عدم دونية: يجوز وصف الاستدلال بالنوافذ كطريقة استخدام صالحة لتغطية الفيديو كاملاً."
        else:
            v = ("لم يتحقق عدم الدونية: النموذج يجب أن يُستخدم كما دُرِّب (30 إطاراً موزعة على الفيديو)، "
                 "والاستدلال بالنوافذ عمل مستقبلي يحتاج تدريباً بنفس الطريقة.")
        lines.append(f"- Δτ = {dm:+.4f}، فترة الثقة [{lo:+.4f}، {hi:+.4f}]، الهامش −{NI_MARGIN}{tag}")
        lines.append(f"- **{v}**")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT, 'windowed_analysis.md'), 'w', encoding='utf-8') as f:
        f.write(text + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    if '--analyze_only' not in sys.argv:
        for ds in DATASETS:
            print(f"\n########## الاستدلال بالنوافذ: {ds} ##########", flush=True)
            evaluate_dataset(ds)
    analyze()
    print(f"\n[✓] انتهى. التحليل في: {os.path.join(OUT, 'windowed_analysis.md')}")


if __name__ == '__main__':
    main()
