"""
Grad-CAM على الإطارات التي يعدّها البشر الأهم — قاعدة اختيار مستقلة عن النموذج، تُكتب في الورقة كما هي.
في PyCharm: زر الفأرة الأيمن ← Run. المدة: أقل من دقيقة. المخرجات في results_qualitative_human/

لكل dataset (نفس الفيديو في results_qualitative_best: صاحب أعلى τ للنموذج، seed 42):
  - الإطارات: أعلى 3 إطارات من الثلاثين في متوسط درجات المقيّمين (item['target'])، مع تخطّي الإطارات شبه
    الموحّدة (انحراف معياري للإضاءة < 10 على 0–255). التعادل يُحسم بترتيب الإطار في الفيديو.
  - لكل إطار: درجة النموذج (بعد sigmoid) وترتيبه بين الثلاثين حسب النموذج، و Grad-CAM بتدرّج درجة ذلك الإطار.
  - fig_gradcam_human3_<ds>.png : صف الإطارات + صف خرائط Grad-CAM.
  - qualitative_human.json      : الإطارات المختارة، درجاتها البشرية، درجة النموذج وترتيبه، وأعلى 3 عند النموذج للمقارنة.
"""
import os
import json

import numpy as np
import torch

import make_qualitative_figures as Q          # نفس الدوال والألوان والإعدادات
from make_qualitative_figures import C, plt, INK, MUTED

OUT = os.path.join(C.WORK_DIR, 'results_qualitative_human')
BEST_JSON = os.path.join(C.WORK_DIR, 'results_qualitative_best', 'qualitative_best.json')
TOP_FRAMES = 3
UNIFORM_STD = 10.0


def frame_std(frames_u8, t):
    return float(frames_u8[t].float().mean(dim=0).std())


def gradcam_for(model, frames, targets):
    """تمريرة أمامية واحدة، ثم Grad-CAM مستقل لكل إطار في targets."""
    store = {}
    h = model.net.backbone.register_forward_hook(lambda _, __, out: store.__setitem__('feat', out))
    with torch.enable_grad(), torch.amp.autocast('cuda', enabled=C.DEVICE.type == 'cuda'):  # كما في evaluate_cv.py
        logits = model(frames.unsqueeze(0).to(C.DEVICE)).float().view(-1)
    h.remove()
    cams = []
    for j, t in enumerate(targets):
        grad = torch.autograd.grad(logits[t], store['feat'], retain_graph=j < len(targets) - 1)[0]
        feat, g = store['feat'][t].detach().float(), grad[t].detach().float()          # (C, h, w)
        cam = torch.relu((g.mean(dim=(1, 2))[:, None, None] * feat).sum(0)).cpu().numpy()
        cams.append(cam / cam.max() if cam.max() > 0 else cam)
    return logits.detach().cpu(), cams


def fig_row(frames, cams, labels, title, path):
    import cv2
    n = len(frames)
    ratio = frames[0].shape[0] / frames[0].shape[1]
    fig, axes = plt.subplots(2, n, figsize=(7.2, 7.2 / n * ratio * 2 + 0.9))
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
        axes[0, c].set_title(lab, fontsize=8, color=INK, linespacing=1.3)
    axes[0, 0].set_ylabel('Frame', fontsize=9, color=INK)
    axes[1, 0].set_ylabel('Grad-CAM', fontsize=9, color=INK)
    fig.suptitle(title, fontsize=9, color=MUTED, y=0.01)
    fig.tight_layout(h_pad=0.4, w_pad=0.4)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    best = json.load(open(BEST_JSON, encoding='utf-8'))
    report = {}
    for ds in ('summe', 'tvsum'):
        name = best[ds]['video']
        cache = C.load_cache(ds)
        names = [str(it['name']) for it in cache]
        idx = names.index(name)
        item = cache[idx]
        k = Q.fold_of(len(cache), idx)
        model, ckpt = Q.load_model(ds, k)
        frames = C.frames_tensor(item)

        human = item['target'].numpy().astype(float)                  # متوسط المقيّمين عند الإطارات الثلاثين
        h_rank = np.argsort(-human, kind='stable')
        chosen, skipped = [], []
        for t in h_rank:
            if frame_std(item['frames_u8'], int(t)) < UNIFORM_STD:
                skipped.append(int(t) + 1)
                continue
            chosen.append(int(t))
            if len(chosen) == TOP_FRAMES:
                break

        logits, cams = gradcam_for(model, frames, chosen)
        m_rank = np.argsort(-logits.numpy(), kind='stable').tolist()
        n = int(item['n_frames'])
        imgs, labels, rows = [], [], []
        for j, t in enumerate(chosen):
            fi = int(item['frame_indices'][t])
            fallback = item['frames_u8'][t].permute(1, 2, 0).numpy()
            imgs.append(Q.native_frame(ds, name, fi, fallback))
            prob = float(torch.sigmoid(logits[t]))
            pct = 100 * fi / max(n - 1, 1)
            mr = m_rank.index(t) + 1
            rows.append(dict(sample=t + 1, human_rank=int(np.where(h_rank == t)[0][0]) + 1, human_score=float(human[t]),
                             video_frame=fi, time_pct=round(pct, 1), model_score=prob, model_rank=mr))
            labels.append(f"human rank {j + 1} · {pct:.0f}% of video\nmodel score {prob:.2f} · model rank {mr}/30")
        fig_row(imgs, cams, labels, f"{name}: the {TOP_FRAMES} sampled frames with the highest human score",
                os.path.join(OUT, f"fig_gradcam_human3_{ds}.png"))

        model_top = [dict(sample=t + 1, time_pct=round(100 * int(item['frame_indices'][t]) / max(n - 1, 1), 1),
                          model_score=float(torch.sigmoid(logits[t])), human_score=float(human[t]),
                          uniform=frame_std(item['frames_u8'], t) < UNIFORM_STD) for t in m_rank[:TOP_FRAMES]]
        report[ds] = dict(rule='3 sampled frames with the highest mean annotator score (near-uniform frames skipped)',
                          video=name, fold=k, checkpoint=os.path.relpath(ckpt, C.WORK_DIR), seed=Q.SEED,
                          frames=rows, skipped_uniform=skipped, model_top3=model_top,
                          model_rank_of_human_top=[r['model_rank'] for r in rows])
        print(f"[{ds}] {name}: human top = {[r['sample'] for r in rows]} | "
              f"ترتيبها عند النموذج = {[r['model_rank'] for r in rows]} | مستبعدة = {skipped}")
    with open(os.path.join(OUT, 'qualitative_human.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[✓] {OUT}")


if __name__ == '__main__':
    main()
