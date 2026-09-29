"""
تقييم موحّد لكل التجارب: نفس الـ pipeline ونفس البروتوكول لكل صفوف الجداول.

لكل فيديو في fold الاختبار يحسب:
  - F1 بالبروتوكولين (strict و standard)، و F1 للنموذج معكوساً (strict)
  - τ و ρ مقابل متوسط المُقيِّمين على 30 نقطة (تعريف الورقة الحالي، للاستمرارية)
  - τ و ρ مقابل كل مُقيِّم على حدة على 150 نقطة (طريقة Otani؛ TVSum فقط)
  - زمن الاستدلال بـ CUDA events (الوسيط من عدة تكرارات) وذروة الذاكرة

ويحسب مرة واحدة لكل مجموعة بيانات: الملخص العشوائي والسقف البشري (results/baselines_<dataset>.csv)
ومرة واحدة لكل نموذج: عدد المعاملات، الحجم، و GFLOPs (results/model_stats.csv)

مثال:
  python evaluate_cv.py --dataset tvsum --backbone mobilevitv2_050 --config proposed --seed 42
  python evaluate_cv.py --dataset tvsum --baselines_only --time_decode
"""
import os
import time
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import common as C
import eval_protocols as E

TIMING_WARMUP = 10
TIMING_REPS = 20


def capacity():
    return max(1, int(C.CAPACITY_RATIO * C.FINE_FRAMES))


# ------------------------------------------------------------
# Baselines (مستقلة عن أي نموذج)
# ------------------------------------------------------------
def compute_baselines(dataset, cache, time_decode=False):
    path = os.path.join(C.RESULTS_DIR, f"baselines_{dataset}.csv")
    if os.path.exists(path) and not time_decode:
        return pd.read_csv(path)

    how = C.HPARAMS[dataset]['reduction']
    rng = np.random.default_rng(0)
    folds = C.get_folds(len(cache))
    fold_of = {int(i): k for k, (_, te) in enumerate(folds, start=1) for i in te}

    rows = []
    for i, item in enumerate(cache):
        ann = item['annotators'].numpy()
        refs = E.references(ann, item['shots'], capacity())
        rnd = E.random_f1(refs, item['shots'], capacity(), C.FINE_FRAMES, how, rng)
        hum = E.human_f1(refs, how)
        row = dict(video=item['name'], fold=fold_of[i],
                   random_f1_strict=rnd['strict'], random_f1_standard=rnd['standard'],
                   human_f1_strict=hum['strict'], human_f1_standard=hum['standard'],
                   duration_s=item['duration_s'], n_frames=item['n_frames'], fps=item['fps'])
        if dataset == 'tvsum':
            row['human_tau_ann'], row['human_rho_ann'] = E.human_rank_corr(ann)
        if time_decode:
            t0 = time.perf_counter()
            C.read_frames(C.video_file(dataset, item['name']), item['frame_indices'])  # seek + decode + resize
            row['decode_ms'] = (time.perf_counter() - t0) * 1000
        rows.append(row)
        print(f"[baselines] {dataset} {i + 1}/{len(cache)} {item['name']}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    df.to_csv(path, index=False)
    return df


# ------------------------------------------------------------
# إحصاءات النموذج (مستقلة عن البيانات)
# ------------------------------------------------------------
def model_stats(backbone, config):
    path = os.path.join(C.RESULTS_DIR, "model_stats.csv")
    df = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    if len(df) and ((df.backbone == backbone) & (df.config == config)).any():
        return

    # train() يعطّل المسار المدمج (fast path) في TransformerEncoderLayer حتى تُحتسب كل العمليات
    model = C.build_model(backbone, config, pretrained=False).to(C.DEVICE).train()
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(1, C.NUM_FRAMES, 3, 224, 224, device=C.DEVICE)
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model(x)
        gflops = fc.get_total_flops() / 1e9
    except Exception as e:  # torch قديم
        print(f"[!] FLOPs غير متاحة: {e}")
        gflops = float('nan')

    row = dict(backbone=backbone, config=config, params=n_params,
               size_fp32_mb=n_params * 4 / 2 ** 20, size_fp16_mb=n_params * 2 / 2 ** 20,
               gflops_per_video=gflops, gflops_per_frame=gflops / C.NUM_FRAMES)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    df.to_csv(path, index=False)


# ------------------------------------------------------------
# التوقيت
# ------------------------------------------------------------
@torch.no_grad()
def forward(model, frames):
    with torch.amp.autocast('cuda', enabled=C.DEVICE.type == 'cuda'):
        return model(frames)


@torch.no_grad()
def timed_forward(model, frames):
    if C.DEVICE.type != 'cuda':
        t0 = time.perf_counter()
        out = forward(model, frames)
        return out, (time.perf_counter() - t0) * 1000, float('nan')
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(TIMING_REPS):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        out = forward(model, frames)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    peak_mb = torch.cuda.max_memory_allocated() / 2 ** 20
    return out, float(np.median(times)), peak_mb


# ------------------------------------------------------------
# تقييم تشغيل واحد (dataset × backbone × config × seed)
# ------------------------------------------------------------
def evaluate_run(dataset, backbone, config, seed, cache, tag=''):
    rdir = C.run_dir(dataset, backbone, config, seed, tag)
    out_dir = os.path.join(C.RESULTS_DIR, tag) if tag else C.RESULTS_DIR  # نتائج الاختبار لا تدخل الملخص
    out_path = os.path.join(out_dir, f"per_video_{dataset}_{backbone}_{config}_seed{seed}.csv")
    how = C.HPARAMS[dataset]['reduction']
    folds = C.get_folds(len(cache))
    rows = []

    for k, (_, test_idx) in enumerate(folds, start=1):
        ckpt = os.path.join(rdir, f"fold_{k}_last.pth")
        if not os.path.exists(ckpt):
            print(f"[!] غير موجود: {ckpt} — تخطّي fold {k}")
            continue
        model = C.build_model(backbone, config, pretrained=False).to(C.DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=C.DEVICE, weights_only=True), strict=True)
        model.eval()

        warm = torch.randn(1, C.NUM_FRAMES, 3, 224, 224, device=C.DEVICE)
        for _ in range(TIMING_WARMUP):
            forward(model, warm)
        if C.DEVICE.type == 'cuda':
            torch.cuda.synchronize()

        for i in test_idx:
            item = cache[i]
            frames = C.frames_tensor(item).unsqueeze(0).to(C.DEVICE)
            logits, latency_ms, peak_mb = timed_forward(model, frames)
            pred30 = logits.float().view(-1).cpu()
            pred150 = F.interpolate(pred30.view(1, 1, -1), size=C.FINE_FRAMES,
                                    mode='linear', align_corners=False).view(-1).numpy()
            ann = item['annotators'].numpy()
            shots = item['shots']
            refs = E.references(ann, shots, capacity())

            f1 = {p: E.f1_vs_refs(E.summary(pred150, shots, capacity(), p), refs[p], how) for p in E.PROTOCOLS}
            f1_inv = E.f1_vs_refs(E.summary(-pred150, shots, capacity(), 'strict'), refs['strict'], how)
            tau30, rho30 = E.rank_corr(pred30.numpy(), item['target'].numpy())
            tau_ann, rho_ann = E.rank_corr_annotators(pred150, ann, per_annotator=(dataset == 'tvsum'))

            rows.append(dict(dataset=dataset, backbone=backbone, config=config, seed=seed, fold=k,
                             video=item['name'], f1_strict=f1['strict'], f1_standard=f1['standard'],
                             f1_strict_inverted=f1_inv, tau_meangt30=tau30, rho_meangt30=rho30,
                             tau_ann=tau_ann, rho_ann=rho_ann, latency_ms=latency_ms, peak_mem_mb=peak_mb))
        fold_df = pd.DataFrame([r for r in rows if r['fold'] == k])
        print(f"  fold {k}: F1 strict={fold_df.f1_strict.mean() * 100:.2f} | "
              f"F1 standard={fold_df.f1_standard.mean() * 100:.2f} | τ_ann={fold_df.tau_ann.mean():.4f} | "
              f"latency={fold_df.latency_ms.median():.2f} ms", flush=True)

    if rows:
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"[✓] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=C.DATASETS)
    ap.add_argument('--backbone', choices=C.ALL_BACKBONES)
    ap.add_argument('--config', choices=list(C.CONFIGS))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--baselines_only', action='store_true')
    ap.add_argument('--time_decode', action='store_true', help='قياس زمن فك ترميز 30 إطاراً لكل فيديو (بطيء)')
    ap.add_argument('--tag', default='', help='نفس الـ tag المستخدم في train_cv.py')
    args = ap.parse_args()

    cache = C.load_cache(args.dataset)
    compute_baselines(args.dataset, cache, time_decode=args.time_decode)
    if args.baselines_only:
        return
    if not (args.backbone and args.config):
        ap.error('--backbone و --config مطلوبان')
    model_stats(args.backbone, args.config)
    print(f"\n=== EVAL {args.dataset} | {args.backbone} | {args.config} | seed {args.seed} ===")
    evaluate_run(args.dataset, args.backbone, args.config, args.seed, cache, args.tag)


if __name__ == '__main__':
    main()
