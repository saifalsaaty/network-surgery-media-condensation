"""
إعادة تقييم F1 بالبروتوكول الرسمي للأدبيات (ملفات h5 الموحّدة: eccv16_dataset_*_google_pool5.h5)
لجعل F1 قابلاً للمقارنة مع الطرق المنشورة. بلا أي تدريب: نفس الـ checkpoints (آخر epoch، 3 seeds، 5 folds).
البند 11 في ANALYSIS_PLAN.md (تحليل لمواءمة البروتوكول فقط؛ لا يُتخذ عليه أي قرار).

البروتوكول الرسمي (vsum_tools.py في KaiyangZhou/pytorch-vsumm-reinforce، مطابق حرفياً):
  - درجات النموذج تُنقل إلى مواقع picks (إطار كل 15 تقريباً)، ثم تُمدّ على الإطارات
  - قيمة اللقطة = متوسط درجات إطاراتها، ولقطات change_points الرسمية
  - knapsack بسعة floor(15% × n_frames)
  - F1 مقابل user_summary: SumMe = max، TVSum = avg

التجهيز (مرة واحدة):
  1) نزّل datasets.tar.gz من رابط Google Drive في README مستودع KaiyangZhou/pytorch-vsumm-reinforce
     وضع الملفين في Unified_Experiments/h5/ :
       eccv16_dataset_summe_google_pool5.h5
       eccv16_dataset_tvsum_google_pool5.h5
  2) إن لم تكن h5py مثبتة: في Terminal داخل PyCharm (بيئة المشروع):  pip install h5py
في PyCharm: زر الفأرة الأيمن ← Run 'eval_official_f1'. المدة المتوقعة: 10–25 دقيقة.
المخرجات: results_official/official_f1_per_video.csv و results_official/official_f1_summary.md
"""
import os
import sys
import math

import numpy as np
import pandas as pd
import torch

os.environ.pop('VS_NUM_FRAMES', None)  # checkpoints و cache الـ 30 إطاراً
import common as C

try:
    import h5py
except ImportError:
    sys.exit("[!] h5py غير مثبتة. في Terminal داخل PyCharm: pip install h5py")

H5 = {'summe': os.path.join(C.WORK_DIR, 'h5', 'eccv16_dataset_summe_google_pool5.h5'),
      'tvsum': os.path.join(C.WORK_DIR, 'h5', 'eccv16_dataset_tvsum_google_pool5.h5')}
METRIC = {'summe': 'max', 'tvsum': 'avg'}
OUT = os.path.join(C.WORK_DIR, 'results_official')
RUNS = ([(bb, cfg) for bb in C.BACKBONES + C.CNN_ABLATION_BACKBONES for cfg in C.CONFIGS] +
        [(bb, 'full_temporal') for bb in C.REFERENCE_BACKBONES])


# ------------------------------------------------------------
# البروتوكول الرسمي (منقول من vsum_tools.py و knapsack.py، Python 3، نفس المنطق)
# ------------------------------------------------------------
def knapsack_dp(values, weights, n_items, capacity):
    table = np.zeros((n_items + 1, capacity + 1), dtype=np.float32)
    keep = np.zeros((n_items + 1, capacity + 1), dtype=bool)
    for i in range(1, n_items + 1):
        wi, vi = int(weights[i - 1]), np.float32(values[i - 1])
        prev = table[i - 1]
        cand = np.full(capacity + 1, -np.inf, dtype=np.float32)
        if wi <= capacity:
            cand[wi:] = prev[:capacity + 1 - wi] + vi
        k = cand > prev  # نفس الشرط الأصلي: (wi <= w) and (vi + table[i-1, w-wi] > table[i-1, w])
        keep[i] = k
        table[i] = np.where(k, cand, prev)
    picks, K = [], capacity
    for i in range(n_items, 0, -1):
        if keep[i, K]:
            picks.append(i - 1)
            K -= int(weights[i - 1])
    return sorted(picks)


def generate_summary(ypred, cps, n_frames, nfps, positions, proportion=0.15):
    frame_scores = np.zeros(n_frames, dtype=np.float32)
    positions = positions.astype(np.int64)
    if positions[-1] != n_frames:
        positions = np.concatenate([positions, [n_frames]])
    for i in range(len(positions) - 1):
        frame_scores[positions[i]:positions[i + 1]] = 0 if i == len(ypred) else ypred[i]
    seg_score = [float(frame_scores[int(s):int(e) + 1].mean()) for s, e in cps]
    limits = int(math.floor(n_frames * proportion))
    picks = set(knapsack_dp(seg_score, nfps, len(cps), limits))
    return np.concatenate([np.ones(int(nf), np.float32) if i in picks else np.zeros(int(nf), np.float32)
                           for i, nf in enumerate(nfps)])


def evaluate_summary(machine, users, metric):
    machine = (machine > 0).astype(np.float32)
    users = (users > 0).astype(np.float32)
    n_users, n_frames = users.shape
    machine = machine[:n_frames] if len(machine) >= n_frames else np.concatenate(
        [machine, np.zeros(n_frames - len(machine), np.float32)])
    f = []
    for u in users:
        ov = (machine * u).sum()
        p, r = ov / (machine.sum() + 1e-8), ov / (u.sum() + 1e-8)
        f.append(0.0 if p == 0 and r == 0 else 2 * p * r / (p + r))
    return float(np.max(f) if metric == 'max' else np.mean(f))


# ------------------------------------------------------------
# مطابقة فيديوهات h5 مع الـ cache (بعدد الإطارات + ارتباط الـ ground truth)
# ------------------------------------------------------------
def load_h5(ds):
    vids = {}
    with h5py.File(H5[ds], 'r') as f:
        for key in f.keys():
            g = f[key]
            name = g['video_name'][()] if 'video_name' in g else None
            if isinstance(name, bytes):
                name = name.decode()
            vids[key] = dict(name=name, n_frames=int(np.array(g['n_frames'])), picks=np.array(g['picks']),
                             cps=np.array(g['change_points']), nfps=np.array(g['n_frame_per_seg']),
                             user_summary=np.array(g['user_summary']), gtscore=np.array(g['gtscore']))
    return vids


def match_videos(ds, cache, vids):
    mapping, rows = {}, []
    for key, v in vids.items():
        best, best_r = None, -2.0
        n_cand = sum(abs(int(it['n_frames']) - v['n_frames']) <= 2 for it in cache)
        for i, item in enumerate(cache):
            if v['name'] and v['name'].replace(' ', '_').lower() == item['name'].replace(' ', '_').lower():
                best, best_r = i, 1.0
                break
            if abs(int(item['n_frames']) - v['n_frames']) > 2:
                continue
            n = int(item['n_frames'])
            ours = np.interp(v['picks'], np.linspace(0, n - 1, C.FINE_FRAMES), item['annotators'].numpy().mean(0))
            r = np.corrcoef(ours, v['gtscore'])[0, 1] if np.std(ours) > 0 and np.std(v['gtscore']) > 0 else -1
            if r > best_r:
                best, best_r = i, r
        rows.append(dict(dataset=ds, h5_key=key, h5_name=v['name'], n_frames=v['n_frames'],
                         matched=cache[best]['name'] if best is not None else None, gt_corr=best_r,
                         candidates_same_length=n_cand))
        if best is not None:
            mapping[key] = best
    rep = pd.DataFrame(rows)
    # المطابقة مؤكدة إذا كان للفيديو مرشح واحد فقط بنفس عدد الإطارات، أو بالاسم؛ وإلا نشترط ارتباطاً ≥ 0.9
    bad = rep[(rep.matched.isna()) | ((rep.gt_corr < 0.9) & (rep.candidates_same_length > 1))]
    print(f"[{ds}] مطابقة {len(mapping)}/{len(vids)} فيديو؛ مطابقات غير مؤكدة: {len(bad)}")
    if len(bad):
        print(bad.to_string(index=False))
    if len(set(mapping.values())) != len(mapping):
        sys.exit(f"[!] {ds}: فيديو واحد من الـ cache طابق أكثر من فيديو h5. أرسل هذا الناتج.")
    return mapping, rep


# ------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    for ds in C.DATASETS:
        if not os.path.exists(H5[ds]):
            sys.exit(f"[!] الملف غير موجود: {H5[ds]}\n    راجع خطوة التجهيز في أعلى هذا السكربت.")
    rows, base_rows, match_reports = [], [], []
    rng = np.random.default_rng(0)
    for ds in C.DATASETS:
        cache = C.load_cache(ds)
        vids = load_h5(ds)
        mapping, rep = match_videos(ds, cache, vids)
        match_reports.append(rep)
        by_cache = {ci: key for key, ci in mapping.items()}
        folds = C.get_folds(len(cache))
        metric = METRIC[ds]

        # العشوائي والبشري بنفس البروتوكول الرسمي
        for ci, key in by_cache.items():
            v = vids[key]
            rnd = np.mean([evaluate_summary(generate_summary(rng.random(len(v['picks'])), v['cps'], v['n_frames'],
                                                             v['nfps'], v['picks']), v['user_summary'], metric)
                           for _ in range(20)])
            us = v['user_summary']
            hum = np.mean([evaluate_summary(us[i], np.delete(us, i, 0), metric) for i in range(len(us))])
            base_rows.append(dict(dataset=ds, video=cache[ci]['name'], random_f1=rnd, human_f1=hum))

        for bb, cfg in RUNS:
            for s in C.SEEDS:
                for k, (_, test_idx) in enumerate(folds, start=1):
                    ckpt = os.path.join(C.run_dir(ds, bb, cfg, s), f"fold_{k}_last.pth")
                    if not os.path.exists(ckpt):
                        print(f"[!] غير موجود: {ckpt}")
                        continue
                    m = C.build_model(bb, cfg, pretrained=False).to(C.DEVICE)
                    m.load_state_dict(torch.load(ckpt, map_location=C.DEVICE, weights_only=True), strict=True)
                    m.eval()
                    for i in test_idx:
                        if i not in by_cache:
                            continue
                        item, v = cache[i], vids[by_cache[i]]
                        with torch.no_grad():
                            p = m(C.frames_tensor(item).unsqueeze(0).to(C.DEVICE)).float().view(-1).cpu().numpy()
                        # مخرج النموذج logits (يُدرَّب بـ BCE-with-logits)؛ نماذج الأدبيات تُخرج احتمالات [0,1].
                        # القيم السالبة تجعل knapsack الرسمي لا يختار أي لقطة، فنحوّلها بالـ sigmoid أولاً.
                        prob = 1.0 / (1.0 + np.exp(-p))
                        ypred = np.interp(v['picks'], np.asarray(item['frame_indices'], float), prob)
                        summ = generate_summary(ypred, v['cps'], v['n_frames'], v['nfps'], v['picks'])
                        rows.append(dict(dataset=ds, backbone=bb, config=cfg, seed=s, fold=k, video=item['name'],
                                         f1_official=evaluate_summary(summ, v['user_summary'], metric)))
                    del m
                print(f"[{ds}] {bb}/{cfg} seed {s}: تم", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT, 'official_f1_per_video.csv'), index=False)

    pd.concat(match_reports).to_csv(os.path.join(OUT, 'h5_video_matching.csv'), index=False)
    base = pd.DataFrame(base_rows)
    base.to_csv(os.path.join(OUT, 'official_baselines.csv'), index=False)
    df = pd.DataFrame(rows)
    lines = ["# F1 بالبروتوكول الرسمي (ملفات h5، change points الرسمية، SumMe = max، TVSum = avg)\n",
             "mean ± std عبر الـ seeds الثلاثة؛ كل فيديو مُقيَّم من الـ fold الذي كان فيه اختباراً.\n"]
    for ds in C.DATASETS:
        d = df[df.dataset == ds]
        b = base[base.dataset == ds]
        lines += [f"\n## {ds.upper()}\n", "| Backbone | Config | F1 official (%) |", "|---|---|---|"]
        for bb, cfg in RUNS:
            g = d[(d.backbone == bb) & (d.config == cfg)]
            if not len(g):
                continue
            ps = g.groupby('seed').f1_official.mean() * 100
            lines.append(f"| {bb} | {cfg} | {ps.mean():.2f} ± {ps.std(ddof=1) if len(ps) > 1 else 0:.2f} |")
        lines.append(f"| — | Random | {b.random_f1.mean() * 100:.2f} |")
        lines.append(f"| — | Human (leave-one-out) | {b.human_f1.mean() * 100:.2f} |")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT, 'official_f1_summary.md'), 'w', encoding='utf-8') as f:
        f.write(text + "\n")
    print(f"\n[✓] انتهى. النتائج في: {OUT}")


if __name__ == '__main__':
    main()
