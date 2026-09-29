"""
أفضل حالة (best case) للأشكال النوعية — بقاعدة اختيار معلنة تُكتب في الورقة كما هي.
في PyCharm: زر الفأرة الأيمن ← Run. المدة: أقل من دقيقة. المخرجات في results_qualitative_best/

القاعدة (لكل dataset):
  - الفيديو: صاحب أعلى Kendall τ (tau_ann) للنموذج v2-050 proposed، seed 42، في per_video_<ds>_..._seed42.csv
    (اختيار بعد رؤية النتائج — يوصف في الورقة بأنه best case وليس حالة نموذجية).
  - النموذج: أوزان آخر epoch من الـ fold الذي كان الفيديو فيه اختباراً (نفس evaluate_cv.py).
  - Grad-CAM: لأعلى 3 إطارات درجةً من الإطارات الثلاثين، كلٌّ منها بتدرّج درجته هو، بلا أقنعة أو عتبات يدوية.
    تُستبعد من الشكل فقط الإطارات شبه الموحّدة (انحراف معياري لشدة الإضاءة < 10 على مقياس 0–255 في الإطار
    الذي رآه النموذج، مثل إطار أبيض أو أسود في نهاية الفيديو)، وتُسجَّل في JSON مع درجتها حتى تُذكر في النص.

المخرجات:
  - fig_temporal_<ds>.png       : نفس تصميم الشكل الزمني السابق.
  - fig_gradcam_top3_<ds>.png   : صف الإطارات الأصلية + صف خرائط Grad-CAM.
  - qualitative_best.json       : أعلى 5 فيديوهات في τ، المقاييس، فحص التطابق مع per_video_*.csv،
                                  وهل يقع كل إطار من الثلاثة داخل ملخص الإجماع البشري.
"""
import os
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import make_qualitative_figures as Q          # نفس الدوال والألوان والإعدادات
from make_qualitative_figures import C, E, plt, INK, MUTED

OUT = os.path.join(C.WORK_DIR, 'results_qualitative_best')
TOP_FRAMES = 3
UNIFORM_STD = 10.0      # إطار شبه موحّد (أبيض/أسود/لون واحد): لا محتوى بصري يُعرض


def best_video(ds):
    p = os.path.join(C.RESULTS_DIR, f"per_video_{ds}_{Q.BACKBONE}_{Q.CONFIG}_seed{Q.SEED}.csv")
    r = pd.read_csv(p).sort_values('tau_ann', ascending=False, kind='mergesort').reset_index(drop=True)
    top5 = [dict(video=str(v), tau_ann=float(t), f1_strict=float(f1)) for v, t, f1 in
            zip(r.video[:5], r.tau_ann[:5], r.f1_strict[:5])]
    return str(r.video.iloc[0]), top5, len(r)


def frame_std(frames_u8, t):
    """الانحراف المعياري لشدة الإضاءة في الإطار كما دخل النموذج (0–255)."""
    return float(frames_u8[t].float().mean(dim=0).std())


def gradcam_topk(model, frames, k, frames_u8):
    """logits لكل الإطارات + Grad-CAM لأعلى k إطارات ذات محتوى بصري (تمريرة أمامية واحدة، تدرّج مستقل لكل إطار)."""
    store = {}
    h = model.net.backbone.register_forward_hook(lambda _, __, out: store.__setitem__('feat', out))
    with torch.enable_grad(), torch.amp.autocast('cuda', enabled=C.DEVICE.type == 'cuda'):  # كما في evaluate_cv.py
        logits = model(frames.unsqueeze(0).to(C.DEVICE)).float().view(-1)
    h.remove()
    ranked = torch.argsort(logits.detach(), descending=True).tolist()
    order, excluded = [], []
    for t in ranked:
        sd = frame_std(frames_u8, t)
        if sd < UNIFORM_STD:
            excluded.append(dict(sample=t + 1, rank=ranked.index(t) + 1, std=sd,
                                 score=float(torch.sigmoid(logits[t].detach()))))
            continue
        order.append(t)
        if len(order) == k:
            break
    cams = []
    for j, t in enumerate(order):
        grad = torch.autograd.grad(logits[t], store['feat'], retain_graph=j < len(order) - 1)[0]
        feat, g = store['feat'][t].detach().float(), grad[t].detach().float()          # (C, h, w)
        cam = torch.relu((g.mean(dim=(1, 2))[:, None, None] * feat).sum(0)).cpu().numpy()
        cams.append(cam / cam.max() if cam.max() > 0 else cam)
    return logits.detach().cpu(), order, cams, excluded, ranked


def fig_gradcam_row(frames, cams, labels, title, path):
    import cv2
    n = len(frames)
    ratio = frames[0].shape[0] / frames[0].shape[1]
    fig, axes = plt.subplots(2, n, figsize=(7.2, 7.2 / n * ratio * 2 + 0.7))
    for c, (fr, cam, lab) in enumerate(zip(frames, cams, labels)):
        hmap = np.clip(cv2.resize(cam.astype(np.float32), (fr.shape[1], fr.shape[0]), interpolation=cv2.INTER_CUBIC), 0, 1)
        colored = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * hmap), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(fr, 0.55, colored, 0.45, 0)
        for r, im in enumerate([fr, overlay]):
            a = axes[r, c]
            a.imshow(im)
            a.set_xticks([])
            a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
        axes[0, c].set_title(lab, fontsize=8.5, color=INK)
    axes[0, 0].set_ylabel('Frame', fontsize=9, color=INK)
    axes[1, 0].set_ylabel('Grad-CAM', fontsize=9, color=INK)
    fig.suptitle(title, fontsize=9, color=MUTED, y=0.01)
    fig.tight_layout(h_pad=0.4, w_pad=0.4)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    report = {}
    for ds in ('summe', 'tvsum'):
        name, top5, n_videos = best_video(ds)
        cache = C.load_cache(ds)
        names = [str(it['name']) for it in cache]
        idx = names.index(name)
        item = cache[idx]
        k = Q.fold_of(len(cache), idx)
        model, ckpt = Q.load_model(ds, k)
        frames = C.frames_tensor(item)
        logits, order, cams, excluded, ranked = gradcam_topk(model, frames, TOP_FRAMES, item['frames_u8'])

        # نفس خطوات evaluate_cv.py بالضبط
        pred150 = F.interpolate(logits.view(1, 1, -1), size=C.FINE_FRAMES, mode='linear',
                                align_corners=False).view(-1).numpy()
        ann = item['annotators'].numpy()
        shots, how = item['shots'], C.HPARAMS[ds]['reduction']
        refs = E.references(ann, shots, Q.capacity())
        pred_bin = E.summary(pred150, shots, Q.capacity(), 'strict')
        f1 = E.f1_vs_refs(pred_bin, refs['strict'], how)
        tau, rho = E.rank_corr_annotators(pred150, ann, per_annotator=(ds == 'tvsum'))
        cons_bin = E.summary(ann.mean(0), shots, Q.capacity(), 'strict')
        prob150 = 1.0 / (1.0 + np.exp(-pred150))
        metrics = dict(f1_strict=f1, tau_ann=tau, rho_ann=rho)
        Q.fig_temporal(ds, name, prob150, item['frame_indices'], item['n_frames'], pred_bin, cons_bin, shots,
                       ann.mean(0), metrics, os.path.join(OUT, f"fig_temporal_{ds}.png"))

        n = int(item['n_frames'])
        imgs, labels, top = [], [], []
        for t in order:
            fi = int(item['frame_indices'][t])
            fallback = item['frames_u8'][t].permute(1, 2, 0).numpy()
            imgs.append(Q.native_frame(ds, name, fi, fallback))
            prob = float(torch.sigmoid(logits[t]))
            j = min(C.FINE_FRAMES - 1, int(fi / max(n - 1, 1) * C.FINE_FRAMES))   # نقطة التقييم التي يقع فيها الإطار
            top.append(dict(sample=t + 1, score_rank=ranked.index(t) + 1, video_frame=fi, time_pct=round(100 * fi / max(n - 1, 1), 1), score=prob,
                            in_consensus_summary=bool(cons_bin[j] > 0), in_model_summary=bool(pred_bin[j] > 0)))
            labels.append(f"sample {t + 1}/30 · {100 * fi / max(n - 1, 1):.0f}% · score {prob:.2f}")
        # كل خريطة مطبّعة على أعلى قيمة فيها (كما في Grad-CAM القياسي)
        fig_gradcam_row(imgs, cams, labels, f"{name}: the {TOP_FRAMES} highest-scoring of the 30 sampled frames"
                        + (f" (excluding {len(excluded)} near-uniform frame{'s' if len(excluded) > 1 else ''})" if excluded else ""),
                        os.path.join(OUT, f"fig_gradcam_top3_{ds}.png"))

        ref = Q.per_video_reference(ds, name)
        ok = ref is not None and abs(ref['f1_strict'] - f1) < 0.02 and abs(ref['tau_ann'] - tau) < 0.01
        report[ds] = dict(rule='highest tau_ann of the proposed model (seed 42) — best case, chosen after the results',
                          video=name, rank=1, n_videos=n_videos, top5_by_tau=top5, fold=k,
                          checkpoint=os.path.relpath(ckpt, C.WORK_DIR), seed=Q.SEED, n_frames=n, **metrics,
                          selected_fraction=float(pred_bin.mean()), top_frames=top, excluded_uniform_frames=excluded,
                          uniform_std_threshold=UNIFORM_STD, per_video_csv=ref, match=ok)
        print(f"[{ds}] {name}: fold {k} | F1 = {f1 * 100:.2f}% | τ = {tau:.4f} | "
              f"أعلى الإطارات = {[t + 1 for t in order]} | مستبعدة (شبه موحّدة) = {[e['sample'] for e in excluded]} | مطابقة per_video_*.csv: {'نعم' if ok else 'لا — أرسل هذا الناتج'}")
    with open(os.path.join(OUT, 'qualitative_best.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[✓] {OUT}")


if __name__ == '__main__':
    main()
