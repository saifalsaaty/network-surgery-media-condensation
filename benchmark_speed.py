"""
قياس السرعة والذاكرة النهائي — مصدر Table 5 و Table 6 في الورقة.
⚠️ شغّله بعد انتهاء run_all.py و run_reference.py، والحاسوب بلا أي حِمل آخر (أغلق المتصفح والبرامج الثقيلة).
في PyCharm: زر الفأرة الأيمن ← Run 'benchmark_speed'.

ثلاثة أجزاء:
  speed     : كل النماذج بالتناوب في جلسة واحدة (عدة جولات بترتيب عشوائي)،
              GPU بـ FP16 و FP32، و CPU بعدد خيوط محدد (محاكاة جهاز حافة، الافتراضي 4).
  memory    : ذروة الذاكرة والزمن مقابل عدد الإطارات K، بطريقتين:
              كل الإطارات دفعة واحدة / على دفعات من 30 إطاراً (تدفق). مصدر Table 6.
              كل نقطة تُقاس في عملية منفصلة بعد استدعاء تحمية (لإعادة القياس وحده: run_memory_benchmark.py).
  precision : هل يغيّر FP16 النتائج؟ مقارنة FP32 و FP16 بالـ checkpoints المدرَّبة (proposed، seed 42).

المخرجات في results/: speed_benchmark.csv، memory_vs_frames.csv، precision_check.csv، benchmark_env.txt
"""
import gc
import os
import time
import random
import argparse
import platform

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau

import common as C
import eval_protocols as E

MODELS = [(bb, cfg) for bb in C.BACKBONES + C.CNN_ABLATION_BACKBONES for cfg in C.CONFIGS] + \
         [(bb, 'full_temporal') for bb in C.REFERENCE_BACKBONES]
MEMORY_MODELS = [('mobilevitv2_050', 'proposed'), ('mobilevitv2_050', 'full_temporal'),
                 ('mobilenetv3_small_100', 'proposed'), ('mobilenetv3_small_100', 'pure'),
                 ('resnet50', 'full_temporal')]
K_LIST = [30, 60, 100, 150, 300, 600, 1200, 2400]
CUDA = torch.cuda.is_available()


# ------------------------------------------------------------
# أدوات
# ------------------------------------------------------------
def amp_forward(model, x):
    with torch.amp.autocast('cuda', enabled=CUDA):
        return model(x)


def forward_chunked(model, x, chunk, device):
    """
    نفس حسابات Summarizer + EdgeVideoSummarizer (stride=1) تماماً، لكن الـ backbone يعالج
    الإطارات على دفعات. x يمكن أن يبقى على الـ CPU وتُنقل كل دفعة وحدها (تدفق).
    """
    net = model.net
    feats = []
    for s in range(0, x.shape[1], chunk):
        xc = x[:, s:s + chunk].to(device, non_blocking=True)
        xc = (xc - model.mean) / model.std
        f = net.backbone(xc.reshape(-1, *xc.shape[2:]))
        if f.dim() > 2:
            f = f.mean(dim=[-2, -1])
        feats.append(f)
    f = net.projection(torch.cat(feats, 0)).view(x.shape[0], x.shape[1], -1)
    f = net.pos_encoder(f)
    for block in net.blocks:
        f = block(f)
    return net.pred_head(f).squeeze(-1)


def gpu_ms(fn, reps):
    times = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return float(np.median(times))


def cpu_ms(fn, reps):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def is_oom(err):
    return isinstance(err, getattr(torch.cuda, 'OutOfMemoryError', ())) or 'out of memory' in str(err).lower()


def flops_table():
    path = os.path.join(C.RESULTS_DIR, 'model_stats.csv')
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=['backbone', 'config'])


# ------------------------------------------------------------
# 1) السرعة
# ------------------------------------------------------------
def part_speed(args):
    models = {}
    for key in MODELS:
        try:
            models[key] = C.build_model(*key, pretrained=False).eval()
        except Exception as ex:
            print(f"[!] تخطّي {key}: {ex}")
    x = torch.rand(1, C.NUM_FRAMES, 3, C.IMAGE_SIZE, C.IMAGE_SIZE)
    res = {k: dict(backbone=k[0], config=k[1], params=sum(p.numel() for p in m.parameters()),
                   gpu_fp16=[], gpu_fp32=[], cpu_fp32=[], peak_gpu_mb=float('nan'))
           for k, m in models.items()}
    rng = random.Random(0)
    keys = list(models)

    if CUDA:
        xg = x.cuda()
        for r in range(args.rounds):
            rng.shuffle(keys)
            print(f"[speed] GPU round {r + 1}/{args.rounds}", flush=True)
            for k in keys:
                m = models[k].cuda()
                with torch.no_grad():
                    for _ in range(args.warmup):
                        amp_forward(m, xg)
                    torch.cuda.synchronize()
                    res[k]['gpu_fp16'].append(gpu_ms(lambda: amp_forward(m, xg), args.reps))
                    for _ in range(args.warmup):
                        m(xg)
                    torch.cuda.synchronize()
                    res[k]['gpu_fp32'].append(gpu_ms(lambda: m(xg), args.reps))
                    if r == 0:
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        amp_forward(m, xg)
                        torch.cuda.synchronize()
                        res[k]['peak_gpu_mb'] = torch.cuda.max_memory_allocated() / 2 ** 20
                models[k] = m.cpu()
                torch.cuda.empty_cache()

    if not args.skip_cpu:
        torch.set_num_threads(args.cpu_threads)
        for r in range(args.cpu_rounds):
            rng.shuffle(keys)
            print(f"[speed] CPU round {r + 1}/{args.cpu_rounds} ({args.cpu_threads} threads)", flush=True)
            for k in keys:
                m = models[k]
                with torch.no_grad():
                    for _ in range(2):
                        m(x)
                    res[k]['cpu_fp32'].append(cpu_ms(lambda: m(x), args.cpu_reps))

    rows = []
    for k, d in res.items():
        row = dict(backbone=d['backbone'], config=d['config'], params=d['params'], peak_gpu_mb=d['peak_gpu_mb'])
        for dev in ('gpu_fp16', 'gpu_fp32', 'cpu_fp32'):
            v = np.array(d[dev]) if d[dev] else np.array([np.nan])
            row[f'{dev}_ms'] = float(np.median(v))
            row[f'{dev}_spread_pct'] = float((v.max() - v.min()) / np.median(v) * 100) if len(v) > 1 else float('nan')
        row['gpu_fp16_frames_per_s'] = C.NUM_FRAMES / (row['gpu_fp16_ms'] / 1000)
        row['cpu_frames_per_s'] = C.NUM_FRAMES / (row['cpu_fp32_ms'] / 1000)
        rows.append(row)
    ft = flops_table()
    df = pd.DataFrame(rows).merge(ft[[c for c in ('backbone', 'config', 'gflops_per_video') if c in ft]],
                                  on=['backbone', 'config'], how='left')
    df.to_csv(os.path.join(C.RESULTS_DIR, 'speed_benchmark.csv'), index=False)
    cols = [c for c in ['backbone', 'config', 'params', 'gflops_per_video', 'gpu_fp16_ms', 'gpu_fp16_spread_pct',
                        'gpu_fp32_ms', 'cpu_fp32_ms', 'peak_gpu_mb', 'gpu_fp16_frames_per_s'] if c in df]
    print("\n### Speed (ms لكل فيديو من 30 إطاراً؛ الوسيط عبر الجولات)\n")
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))


# ------------------------------------------------------------
# 2) الذاكرة مقابل عدد الإطارات — كل نقطة في عملية (process) منفصلة
# ------------------------------------------------------------
# لماذا عملية منفصلة؟ في القياس داخل عملية واحدة ظهرت قراءات غير متسقة، لأن:
#   (1) cuDNN autotuning يحجز ذاكرة عمل مؤقتة في أول استدعاء لكل حجم إدخال جديد فتدخل في الذروة؛
#   (2) ذاكرة الـ allocator المتبقية من النقاط السابقة تؤثر في اللاحقة.
# الحل: عملية جديدة لكل نقطة، واستدعاء تحمية أولاً، ثم تصفير الذروة، ثم القياس.
# الذروة = الأوزان + الإدخال الموجود على الـ GPU + الـ activations (FP16 autocast، cudnn.benchmark=True كما في جزء السرعة).
MEMORY_MODES = ('all_at_once', 'chunked_30')


def measure_memory_point(bb, cfg, K, mode):
    """تُستدعى داخل العملية الابنة فقط."""
    torch.backends.cudnn.benchmark = True
    vram_mb = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
    row = dict(backbone=bb, config=cfg, frames=K, mode=mode, vram_mb=vram_mb)
    m = C.build_model(bb, cfg, pretrained=False).eval().cuda()
    x = fn = None
    try:
        with torch.no_grad():
            if mode == 'all_at_once':
                x = torch.rand(1, K, 3, C.IMAGE_SIZE, C.IMAGE_SIZE, device='cuda')
                fn = lambda: amp_forward(m, x)
            else:
                x = torch.rand(1, K, 3, C.IMAGE_SIZE, C.IMAGE_SIZE).pin_memory()
                fn = lambda: forward_chunked_amp(m, x)
            fn()                                   # تحمية: يجري فيها cuDNN autotuning
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            fn()                                   # الاستدعاء المقاس
            torch.cuda.synchronize()
            row['peak_mb'] = torch.cuda.max_memory_allocated() / 2 ** 20
            row['exceeds_vram'] = bool(row['peak_mb'] > vram_mb)
            # بعد تجاوز ذاكرة الـ GPU يستخدم Windows ذاكرة النظام المشتركة، فيصبح الزمن بلا معنى
            row['latency_ms'] = float('nan') if row['exceeds_vram'] else gpu_ms(fn, 5)
            row['status'] = 'ok_spilled_to_shared_memory' if row['exceeds_vram'] else 'ok'
            if K == C.NUM_FRAMES and mode == 'all_at_once':
                # فحص تطابق: التنفيذ على دفعات يعطي نفس المخرجات (FP32)
                xs = x[:, :C.NUM_FRAMES]
                row['chunk_check_max_diff'] = (m(xs) - forward_chunked(m, xs, 7, 'cuda')).abs().max().item()
    except Exception as ex:
        if not is_oom(ex):
            raise
        row.update(peak_mb=float('nan'), latency_ms=float('nan'), exceeds_vram=True, status='OOM')
    return row


def part_memory(args):
    import sys
    import json
    import subprocess
    if not CUDA:
        print("[memory] يحتاج GPU — تخطّي")
        return
    out_path = os.path.join(C.RESULTS_DIR, 'memory_vs_frames.csv')
    old = os.path.join(C.RESULTS_DIR, 'memory_vs_frames_single_process_OLD.csv')
    if os.path.exists(out_path) and not os.path.exists(old):
        os.replace(out_path, old)  # نحتفظ بالقياس القديم للمقارنة فقط
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    here = os.path.dirname(os.path.abspath(__file__))
    rows = []
    for bb, cfg in MEMORY_MODELS:
        oom_at = None
        for K in K_LIST:
            for mode in MEMORY_MODES:
                if mode == 'all_at_once' and oom_at is not None:
                    row = dict(backbone=bb, config=cfg, frames=K, mode=mode, status=f'skipped (OOM at K={oom_at})')
                else:
                    r = subprocess.run([sys.executable, os.path.abspath(__file__), '--one_memory', bb, cfg, str(K), mode],
                                       cwd=here, env=env, capture_output=True, text=True, encoding='utf-8',
                                       errors='replace')
                    res = [l for l in r.stdout.splitlines() if l.startswith('RESULT ')]
                    if res:
                        row = json.loads(res[-1][len('RESULT '):])
                    else:
                        row = dict(backbone=bb, config=cfg, frames=K, mode=mode, status=f'crash (exit {r.returncode})')
                        print(r.stderr[-1500:])
                    if mode == 'all_at_once' and row.get('status') == 'OOM':
                        oom_at = K
                rows.append(row)
                pd.DataFrame(rows).to_csv(out_path, index=False)  # حفظ تدريجي
                print(f"  {bb}/{cfg}  K={K:5d} {mode:12s} → {row.get('status')}  "
                      f"peak={row.get('peak_mb', float('nan')):.0f} MB  "
                      f"latency={row.get('latency_ms', float('nan')):.1f} ms", flush=True)
    print(f"\n[✓] {out_path}")


def forward_chunked_amp(model, x):
    with torch.amp.autocast('cuda', enabled=CUDA):
        return forward_chunked(model, x, C.NUM_FRAMES, 'cuda')


# ------------------------------------------------------------
# 3) FP16 مقابل FP32
# ------------------------------------------------------------
def part_precision(args):
    if not CUDA:
        print("[precision] يحتاج GPU — تخطّي")
        return
    cap = max(1, int(C.CAPACITY_RATIO * C.FINE_FRAMES))
    rows = []
    for ds in C.DATASETS:
        cache = C.load_cache(ds)
        folds = C.get_folds(len(cache))
        how = C.HPARAMS[ds]['reduction']
        for bb in C.BACKBONES + C.CNN_ABLATION_BACKBONES:
            rdir = C.run_dir(ds, bb, 'proposed', 42)
            for k, (_, test_idx) in enumerate(folds, start=1):
                ckpt = os.path.join(rdir, f"fold_{k}_last.pth")
                if not os.path.exists(ckpt):
                    continue
                m = C.build_model(bb, 'proposed', pretrained=False).cuda()
                m.load_state_dict(torch.load(ckpt, map_location='cuda', weights_only=True))
                m.eval()
                for i in test_idx:
                    item = cache[i]
                    x = C.frames_tensor(item).unsqueeze(0).cuda()
                    with torch.no_grad():
                        p32 = m(x).float().view(-1).cpu()
                        p16 = amp_forward(m, x).float().view(-1).cpu()
                    ann = item['annotators'].numpy()
                    refs = E.references(ann, item['shots'], cap)
                    out = dict(dataset=ds, backbone=bb, fold=k, video=item['name'],
                               max_abs_score_diff=float((p32 - p16).abs().max()),
                               tau_fp32_vs_fp16=float(kendalltau(p32.numpy(), p16.numpy())[0]))
                    bins = {}
                    for tag, p in (('fp32', p32), ('fp16', p16)):
                        p150 = F.interpolate(p.view(1, 1, -1), size=C.FINE_FRAMES, mode='linear',
                                             align_corners=False).view(-1).numpy()
                        bins[tag] = E.summary(p150, item['shots'], cap, 'strict')
                        out[f'f1_strict_{tag}'] = E.f1_vs_refs(bins[tag], refs['strict'], how)
                        out[f'tau_ann_{tag}'] = E.rank_corr_annotators(p150, ann, per_annotator=(ds == 'tvsum'))[0]
                    out['same_summary'] = bool((bins['fp32'] == bins['fp16']).all())
                    rows.append(out)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.RESULTS_DIR, 'precision_check.csv'), index=False)
    if len(df):
        print("\n### FP16 مقابل FP32 (proposed، seed 42)\n")
        print(df.groupby(['dataset', 'backbone']).agg(
            videos=('video', 'count'), same_summary_pct=('same_summary', lambda s: 100 * s.mean()),
            f1_fp32=('f1_strict_fp32', 'mean'), f1_fp16=('f1_strict_fp16', 'mean'),
            tau_fp32=('tau_ann_fp32', 'mean'), tau_fp16=('tau_ann_fp16', 'mean'),
            max_score_diff=('max_abs_score_diff', 'max')).to_string(float_format=lambda v: f"{v:.4f}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', nargs='*', default=['speed', 'memory', 'precision'],
                    choices=['speed', 'memory', 'precision'])
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--reps', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--cpu_threads', type=int, default=4, help='محاكاة معالج جهاز حافة (مثل Jetson: 4-6 أنوية)')
    ap.add_argument('--cpu_rounds', type=int, default=3)
    ap.add_argument('--cpu_reps', type=int, default=5)
    ap.add_argument('--skip_cpu', action='store_true')
    ap.add_argument('--one_memory', nargs=4, metavar=('BACKBONE', 'CONFIG', 'K', 'MODE'),
                    help='داخلي: قياس نقطة ذاكرة واحدة في هذه العملية (يستدعيه part_memory)')
    args = ap.parse_args()

    if args.one_memory:
        import json
        bb, cfg, K, mode = args.one_memory
        print('RESULT ' + json.dumps(measure_memory_point(bb, cfg, int(K), mode)), flush=True)
        return

    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    if 'speed' in args.parts:  # لا نلمس الـ GPU في العملية الأم عند قياس الذاكرة وحدها
        with open(os.path.join(C.RESULTS_DIR, 'benchmark_env.txt'), 'w', encoding='utf-8') as f:
            f.write(f"GPU: {torch.cuda.get_device_name(0) if CUDA else 'none'}\n"
                    f"CPU: {platform.processor()} | logical cores: {os.cpu_count()} | threads used: {args.cpu_threads}\n"
                    f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda} | OS: {platform.platform()}\n")
    torch.backends.cudnn.benchmark = True  # للقياس فقط (لا تدريب هنا)

    if 'speed' in args.parts:
        part_speed(args)
    if 'memory' in args.parts:
        part_memory(args)
    if 'precision' in args.parts:
        part_precision(args)
    print("\n[✓] انتهى القياس. النتائج في مجلد results.")


if __name__ == '__main__':
    main()
