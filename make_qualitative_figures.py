"""
الأشكال النوعية للورقة من النموذج النهائي نفسه (بدل الأشكال القديمة المأخوذة من fold_5_best و100 إطار).
في PyCharm: زر الفأرة الأيمن ← Run. المدة: أقل من دقيقة. المخرجات في results_qualitative/

لكل فيديو (نفس الفيديوهين في الورقة: Kids_playing_in_leaves من SumMe و gzDbaEs1Rlg من TVSum):
  - النموذج: v2-050 proposed، seed 42، أوزان آخر epoch من الـ fold الذي كان الفيديو فيه اختباراً
    (أي تقييم خارج بيانات التدريب، بلا اختيار).
  - fig_temporal_<ds>.png : درجة أهمية النموذج (30 إطاراً موزعة، ممدودة إلى 150 نقطة تقييم كما في evaluate_cv.py)
                            مقابل متوسط درجات المقيّمين، واللقطات التي اختارها knapsack (15%)، وشريطا الملخص
                            (النموذج مقابل ملخص الإجماع البشري بنفس knapsack).
  - fig_gradcam_<ds>.png  : Grad-CAM على مخرج الـ backbone المقطوع للإطار الأعلى درجة (قاعدة اختيار ثابتة)،
                            بلا أقنعة أو عتبات يدوية.
  - qualitative_metrics.json : F1 و τ لكل فيديو، مع مقارنتها بقيم per_video_*.csv (فحص تطابق).
"""
import os
import json

import numpy as np
import torch
import torch.nn.functional as F

os.environ.pop('VS_NUM_FRAMES', None)
import common as C
import eval_protocols as E

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BACKBONE, CONFIG, SEED = 'mobilevitv2_050', 'proposed', 42
VIDEOS = {'summe': 'Kids_playing_in_leaves', 'tvsum': 'gzDbaEs1Rlg'}
OUT = os.path.join(C.WORK_DIR, 'results_qualitative')
BLUE, GRAY, INK, MUTED = '#2a78d6', '#8a8984', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})


def capacity():
    return max(1, int(C.CAPACITY_RATIO * C.FINE_FRAMES))


def fold_of(n_videos, idx):
    for k, (_, test_idx) in enumerate(C.get_folds(n_videos), start=1):
        if idx in test_idx:
            return k
    raise ValueError(idx)


def load_model(ds, k):
    ckpt = os.path.join(C.run_dir(ds, BACKBONE, CONFIG, SEED), f"fold_{k}_last.pth")
    m = C.build_model(BACKBONE, CONFIG, pretrained=False).to(C.DEVICE)
    m.load_state_dict(torch.load(ckpt, map_location=C.DEVICE, weights_only=True), strict=True)
    return m.eval(), ckpt


def gradcam_all(model, frames):
    """logits لكل الإطارات + Grad-CAM للإطار المطلوب لاحقاً (المخرج المكاني للـ backbone المقطوع)."""
    store = {}

    def hook(_, __, out):
        store['feat'] = out
        out.register_hook(lambda g: store.__setitem__('grad', g))

    h = model.net.backbone.register_forward_hook(hook)
    with torch.enable_grad(), torch.amp.autocast('cuda', enabled=C.DEVICE.type == 'cuda'):  # كما في evaluate_cv.py
        logits = model(frames.unsqueeze(0).to(C.DEVICE)).float().view(-1)
        t = int(torch.argmax(logits).item())
        logits[t].backward()
    h.remove()
    feat, grad = store['feat'][t].detach().float(), store['grad'][t].detach().float()   # (C, h, w)
    w = grad.mean(dim=(1, 2))
    cam = torch.relu((w[:, None, None] * feat).sum(0)).cpu().numpy()
    cam = cam / cam.max() if cam.max() > 0 else cam
    return logits.detach().cpu(), t, cam


def native_frame(ds, name, frame_idx, fallback):
    import cv2
    cap = cv2.VideoCapture(C.video_file(ds, name))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, fr = cap.read()
    cap.release()
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) if ok else fallback


def per_video_reference(ds, name):
    p = os.path.join(C.RESULTS_DIR, f"per_video_{ds}_{BACKBONE}_{CONFIG}_seed{SEED}.csv")
    if not os.path.exists(p):
        return None
    import pandas as pd
    r = pd.read_csv(p)
    r = r[r.video.astype(str) == str(name)]
    return None if r.empty else dict(f1_strict=float(r.f1_strict.iloc[0]), tau_ann=float(r.tau_ann.iloc[0]))


def fig_temporal(ds, name, prob150, idx30, n_frames, pred_bin, cons_bin, shots, human_mean, metrics, path):
    x = np.linspace(0, 100, C.FINE_FRAMES, endpoint=False) + 50.0 / C.FINE_FRAMES
    edges = np.linspace(0, 100, C.FINE_FRAMES + 1)
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(7.2, 3.6), gridspec_kw={'height_ratios': [3, 1.1]}, sharex=True)
    for s, e in shots:                                              # لقطات اختارها النموذج
        if pred_bin[s:e].any():
            ax.axvspan(edges[s], edges[e], color=BLUE, alpha=0.12, lw=0)
    h = (human_mean - human_mean.min()) / (human_mean.max() - human_mean.min() + 1e-8)
    ax.plot(x, h, color=GRAY, lw=1.5, label='Human mean score (min-max)')
    pm = (prob150 - prob150.min()) / (prob150.max() - prob150.min() + 1e-8)  # نفس التطبيع قبل knapsack
    ax.plot(x, pm, color=BLUE, lw=2, label='Model score (min-max)')
    xs = np.asarray(idx30, float) / max(n_frames - 1, 1) * 100
    ax.plot(xs, np.interp(xs, x, pm), 'o', ms=4, color=BLUE, mec='white', mew=1,
            label='30 sampled frames')
    ax.set_ylabel('Normalized score')
    ax.set_ylim(-0.02, 1.32)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis='y', color='#e6e5e1', lw=0.8)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.legend(loc='upper right', fontsize=8, frameon=False, ncol=3)
    ax.set_title(f"{name}  |  F1 = {metrics['f1_strict'] * 100:.1f}%  |  Kendall τ = {metrics['tau_ann']:.3f}",
                 fontsize=10, color=INK, loc='left')
    for row, (b, lab, col) in enumerate([(cons_bin, 'Human consensus', GRAY), (pred_bin, 'Model', BLUE)]):
        on = np.flatnonzero(b)
        bx.broken_barh([(edges[i], edges[i + 1] - edges[i]) for i in on], (1.2 - row, 0.7), facecolors=col)
    bx.set_yticks([1.55, 0.55])
    bx.set_yticklabels(['Human\nconsensus', 'Model'])
    bx.set_ylim(0, 2.1)
    bx.set_xlim(0, 100)
    bx.set_xlabel('Video time (%)')
    for sp in ('top', 'right', 'left'):
        bx.spines[sp].set_visible(False)
    bx.tick_params(axis='y', length=0)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def fig_gradcam(frame, cam, title, path):
    import cv2
    hmap = cv2.resize(cam.astype(np.float32), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
    hmap = np.clip(hmap, 0, 1)
    colored = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * hmap), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(frame, 0.55, colored, 0.45, 0)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 7.2 * frame.shape[0] / frame.shape[1] / 2 + 0.5))
    for a, im, t in zip(axes, [frame, overlay], ['Sampled frame', 'Grad-CAM (truncated backbone)']):
        a.imshow(im)
        a.set_title(t, fontsize=10, color=INK)
        a.axis('off')
    fig.suptitle(title, fontsize=9, color=MUTED, y=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    report = {}
    for ds, name in VIDEOS.items():
        cache = C.load_cache(ds)
        names = [str(it['name']) for it in cache]
        if name not in names:
            raise SystemExit(f"[!] {name} غير موجود في cache {ds}")
        idx = names.index(name)
        item = cache[idx]
        k = fold_of(len(cache), idx)
        model, ckpt = load_model(ds, k)
        frames = C.frames_tensor(item)
        logits, t_star, cam = gradcam_all(model, frames)

        # نفس خطوات evaluate_cv.py بالضبط
        pred150 = F.interpolate(logits.view(1, 1, -1), size=C.FINE_FRAMES, mode='linear',
                                align_corners=False).view(-1).numpy()
        ann = item['annotators'].numpy()
        shots, how = item['shots'], C.HPARAMS[ds]['reduction']
        refs = E.references(ann, shots, capacity())
        pred_bin = E.summary(pred150, shots, capacity(), 'strict')
        f1 = E.f1_vs_refs(pred_bin, refs['strict'], how)
        tau, rho = E.rank_corr_annotators(pred150, ann, per_annotator=(ds == 'tvsum'))
        cons_bin = E.summary(ann.mean(0), shots, capacity(), 'strict')
        prob150 = 1.0 / (1.0 + np.exp(-pred150))
        metrics = dict(f1_strict=f1, tau_ann=tau, rho_ann=rho)

        fig_temporal(ds, name, prob150, item['frame_indices'], item['n_frames'], pred_bin, cons_bin, shots,
                     ann.mean(0), metrics, os.path.join(OUT, f"fig_temporal_{ds}.png"))
        fi = item['frame_indices'][t_star]
        fallback = item['frames_u8'][t_star].permute(1, 2, 0).numpy()
        frame = native_frame(ds, name, fi, fallback)
        prob_t = float(torch.sigmoid(logits[t_star]))
        fig_gradcam(frame, cam, f"{name}: highest-scoring of the 30 sampled frames "
                                f"(sample {t_star + 1}/30, video frame {fi}, score {prob_t:.2f})",
                    os.path.join(OUT, f"fig_gradcam_{ds}.png"))

        ref = per_video_reference(ds, name)
        report[ds] = dict(video=name, fold=k, checkpoint=os.path.relpath(ckpt, C.WORK_DIR), seed=SEED,
                          n_frames=int(item['n_frames']), top_sample=t_star + 1, top_video_frame=int(fi),
                          top_score=prob_t, gradcam_map=list(cam.shape), **metrics,
                          selected_fraction=float(pred_bin.mean()), per_video_csv=ref)
        ok = ref is None or (abs(ref['f1_strict'] - f1) < 0.02 and abs(ref['tau_ann'] - tau) < 0.01)
        print(f"[{ds}] {name}: fold {k} | F1 = {f1 * 100:.2f}% | τ = {tau:.4f} | "
              f"أعلى إطار = {t_star + 1}/30 | مطابقة per_video_*.csv: {'نعم' if ok else 'لا — أرسل هذا الناتج'}")
    with open(os.path.join(OUT, 'qualitative_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[✓] {OUT}")


if __name__ == '__main__':
    main()
