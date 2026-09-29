"""
ذروة ذاكرة RAM على المعالج (بلا GPU) مقابل عدد الإطارات، بطريقتين:
  all_at_once : كل إطارات الفيديو في الذاكرة ثم تمريرة واحدة عبر النموذج
  streaming   : تُقرأ الإطارات 30 إطاراً في كل مرة، وتُحفظ ميزاتها فقط (متجه صغير لكل إطار)،
                ثم يعمل الرأس الزمني على كل الميزات. هذا ما يفعله جهاز حافة يقرأ الفيديو من الملف.

في PyCharm: زر الفأرة الأيمن ← Run 'benchmark_cpu_ram'. المدة المتوقعة: 10–15 دقيقة.
أغلق المتصفح والبرامج الثقيلة قبل التشغيل.
المخرجات: results/cpu_ram_vs_frames.csv و results/cpu_ram_vs_frames.md

طريقة القياس:
  - كل نقطة (نموذج، عدد إطارات، طريقة) في عملية Python جديدة، فلا تتأثر نقطة بما قبلها.
  - نقيس ذاكرة العملية قبل التمرير (بعد تحميل النموذج) ثم ذروتها أثناء التمرير.
    الزيادة = ما تحتاجه الطريقة نفسها (الإطارات + الحسابات الوسيطة)، بلا Python و PyTorch والأوزان.
  - Windows: Peak Working Set (الذاكرة الفعلية) و Peak Private Bytes (ما حُجز فعلاً، لا يتأثر بالـ paging).
  - FP32، 4 خيوط CPU (نفس قياس السرعة)، أوزان عشوائية (الذاكرة لا تتأثر بقيم الأوزان).
  - الإطارات عشوائية بحجم 224×224؛ فك ترميز الفيديو نفسه غير مشمول (ذاكرته ثابتة وصغيرة).
  - أمان: إذا توقّعنا (من النقطة السابقة) أن all_at_once سيحتاج أكثر من 80% من الذاكرة المتاحة،
    لا نشغّله حتى لا يتجمّد الحاسوب، ويُسجَّل "تجاوز متوقَّع" مع الرقم المتوقَّع.
"""
import os
import sys
import gc
import json
import time
import platform
import argparse
import threading
import subprocess

os.environ.pop('VS_NUM_FRAMES', None)  # نفس إعداد الـ 30 إطاراً
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
MB = 2 ** 20
CHUNK = 30
K_LIST = [30, 60, 100, 150, 300, 600, 1200, 2400]
MODES = ('all_at_once', 'streaming')
DEFAULT_MODELS = ['mobilevitv2_050:proposed', 'mobilenetv3_small_100:pure']
SAFETY_FRACTION = 0.80   # لا نشغّل all_at_once إذا توقّعنا تجاوز 80% من الذاكرة المتاحة
GROWTH_MARGIN = 1.15     # هامش على التوقّع الخطي


# ------------------------------------------------------------
# قراءة الذاكرة (Windows عبر ctypes؛ Linux للاختبار فقط)
# ------------------------------------------------------------
if os.name == 'nt':
    import ctypes
    from ctypes import wintypes

    class _PMC(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                    ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]

    class _MSX(ctypes.Structure):
        _fields_ = [('dwLength', wintypes.DWORD), ('dwMemoryLoad', wintypes.DWORD),
                    ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]

    _k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    _psapi = ctypes.WinDLL('psapi', use_last_error=True)
    _k32.GetCurrentProcess.restype = wintypes.HANDLE
    _k32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MSX)]
    _k32.GlobalMemoryStatusEx.restype = wintypes.BOOL
    _psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
    _psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    def mem_now():
        c = _PMC()
        c.cb = ctypes.sizeof(_PMC)
        if not _psapi.GetProcessMemoryInfo(_k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return dict(ws=c.WorkingSetSize, peak_ws=c.PeakWorkingSetSize,
                    private=c.PagefileUsage, peak_private=c.PeakPagefileUsage)

    def system_ram():
        s = _MSX()
        s.dwLength = ctypes.sizeof(_MSX)
        if not _k32.GlobalMemoryStatusEx(ctypes.byref(s)):
            raise ctypes.WinError(ctypes.get_last_error())
        return s.ullTotalPhys, s.ullAvailPhys
else:
    def _status_kb(field):
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith(field + ':'):
                    return int(line.split()[1]) * 1024
        return 0

    def mem_now():
        rss, hwm = _status_kb('VmRSS'), _status_kb('VmHWM')
        return dict(ws=rss, peak_ws=hwm, private=rss, peak_private=hwm)

    def system_ram():
        info = {}
        with open('/proc/meminfo') as f:
            for line in f:
                k, v = line.split(':')
                info[k] = int(v.split()[0]) * 1024
        return info['MemTotal'], info['MemAvailable']


class PeakSampler(threading.Thread):
    """يقرأ ذاكرة العملية كل 2 ms أثناء التمرير (احتياط إذا كانت ذروة سابقة أعلى من ذروة التمرير)."""

    def __init__(self, period=0.002):
        super().__init__(daemon=True)
        self.period, self.running = period, True
        self.max_ws = self.max_private = 0

    def run(self):
        while self.running:
            m = mem_now()
            self.max_ws = max(self.max_ws, m['ws'])
            self.max_private = max(self.max_private, m['private'])
            time.sleep(self.period)


# ------------------------------------------------------------
# نقطة قياس واحدة (داخل العملية الابنة)
# ------------------------------------------------------------
def forward_streaming(model, K, chunk, next_frames):
    """
    نفس حسابات النموذج تماماً (نفس forward_chunked في benchmark_speed.py الذي تحققنا من تطابقه)،
    لكن لا يوجد في الذاكرة إلا `chunk` إطارات في أي لحظة، إضافة إلى ميزات الإطارات السابقة.
    """
    import torch
    net = model.net
    feats = []
    for s in range(0, K, chunk):
        xc = next_frames(s, min(chunk, K - s))  # (1, n, 3, H, W): يحاكي قراءة n إطارات من الفيديو
        xc = (xc - model.mean) / model.std
        f = net.backbone(xc.reshape(-1, *xc.shape[2:]))
        if f.dim() > 2:
            f = f.mean(dim=[-2, -1])
        feats.append(f)
        del xc
    f = net.projection(torch.cat(feats, 0)).view(1, K, -1)
    f = net.pos_encoder(f)
    for block in net.blocks:
        f = block(f)
    return net.pred_head(f).squeeze(-1)


def measure_point(bb, cfg, K, mode, threads):
    import torch
    import common as C
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    S = C.IMAGE_SIZE
    frame_mb = 3 * S * S * 4 / MB
    m = C.build_model(bb, cfg, pretrained=False).eval()
    gc.collect()
    total, avail = system_ram()
    before = mem_now()
    row = dict(backbone=bb, config=cfg, frames=K, mode=mode, threads=threads,
               frames_in_memory_mb=round((K if mode == 'all_at_once' else min(CHUNK, K)) * frame_mb, 1),
               baseline_mb=before['ws'] / MB, total_ram_mb=total / MB, avail_ram_mb=avail / MB)
    sampler = PeakSampler()
    sampler.start()
    t0 = time.perf_counter()
    status = 'ok'
    try:
        with torch.no_grad():
            if mode == 'all_at_once':
                x = torch.rand(1, K, 3, S, S)
                y = m(x)
                del x
            else:
                y = forward_streaming(m, K, CHUNK, lambda s, n: torch.rand(1, n, 3, S, S))
        row['time_s'] = time.perf_counter() - t0
        del y
    except (RuntimeError, MemoryError) as ex:
        msg = str(ex).lower()
        if isinstance(ex, MemoryError) or 'not enough memory' in msg or 'alloc' in msg:
            status = 'OOM'
            row['time_s'] = float('nan')
        else:
            raise
    finally:
        sampler.running = False
        sampler.join()
    after = mem_now()
    # إذا كانت ذروة العملية ارتفعت أثناء التمرير فهي ذروة التمرير بالضبط؛ وإلا نأخذ أعلى قراءة أثناءه
    peak_ws = after['peak_ws'] if after['peak_ws'] > before['peak_ws'] else max(sampler.max_ws, after['ws'])
    peak_pv = (after['peak_private'] if after['peak_private'] > before['peak_private']
               else max(sampler.max_private, after['private']))
    row.update(peak_ram_mb=peak_ws / MB,
               peak_ram_increase_mb=(peak_ws - before['ws']) / MB,
               peak_private_increase_mb=(peak_pv - before['private']) / MB,
               status=status)
    # فحص تطابق مرة واحدة: التمرير على دفعات يعطي نفس المخرجات
    if status == 'ok' and mode == 'all_at_once' and K == CHUNK:
        with torch.no_grad():
            xs = torch.rand(1, K, 3, S, S, generator=torch.Generator().manual_seed(1))
            row['stream_check_max_diff'] = (m(xs) - forward_streaming(
                m, K, 7, lambda s, n: xs[:, s:s + n])).abs().max().item()
    return row


# ------------------------------------------------------------
# التشغيل الكامل (العملية الأم)
# ------------------------------------------------------------
def run_point(bb, cfg, K, mode, threads, timeout):
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    try:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), '--one', bb, cfg, str(K), mode, str(threads)],
                           cwd=HERE, env=env, capture_output=True, text=True, encoding='utf-8',
                           errors='replace', timeout=timeout)
    except subprocess.TimeoutExpired:
        return dict(backbone=bb, config=cfg, frames=K, mode=mode, status=f'timeout (> {timeout} s)')
    res = [l for l in r.stdout.splitlines() if l.startswith('RESULT ')]
    if res:
        return json.loads(res[-1][len('RESULT '):])
    print(r.stderr[-1500:])
    return dict(backbone=bb, config=cfg, frames=K, mode=mode, status=f'crash (exit {r.returncode})')


def write_summary(df, path, threads, total_ram_mb):
    frame_mb = 3 * 224 * 224 * 4 / MB
    if 'frames_in_memory_mb' in df and df['frames_in_memory_mb'].notna().any():
        ok = df[df['frames_in_memory_mb'].notna()].iloc[0]
        frame_mb = ok['frames_in_memory_mb'] / (ok['frames'] if ok['mode'] == 'all_at_once' else min(CHUNK, ok['frames']))
    lines = ["# ذروة RAM على المعالج مقابل عدد الإطارات\n",
             f"المعالج: {platform.processor()} | الذاكرة الكلية: {total_ram_mb / 1024:.1f} GB | خيوط: {threads} | FP32\n",
             "الزيادة = ذروة ذاكرة العملية أثناء التمرير − ذاكرتها قبله (بعد تحميل النموذج)، بالـ MB.\n"]
    for (bb, cfg), g in df.groupby(['backbone', 'config'], sort=False):
        lines += [f"\n## {bb} / {cfg}\n",
                  "| الإطارات | حجم كل الإطارات (MB) | دفعة واحدة | تدفّق 30 إطاراً |",
                  "|---|---|---|---|"]
        for K, gk in g.groupby('frames', sort=True):
            cells = []
            for mode in MODES:
                r = gk[gk['mode'] == mode]
                if not len(r):
                    cells.append('—')
                    continue
                r = r.iloc[0]
                cells.append(f"{r['peak_ram_increase_mb']:.0f}" if r.get('status') == 'ok' else str(r.get('status')))
            lines.append(f"| {K} | {K * frame_mb:.0f} | {cells[0]} | {cells[1]} |")
        if 'stream_check_max_diff' in g and g['stream_check_max_diff'].notna().any():
            d = g['stream_check_max_diff'].dropna().iloc[0]
            lines.append(f"\nتطابق المخرجات (30 إطاراً: دفعة واحدة مقابل دفعات من 7): أكبر فرق = {d:.2e}")
    text = "\n".join(lines) + "\n"
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=DEFAULT_MODELS, help='backbone:config')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--timeout', type=int, default=1800, help='أقصى زمن لكل نقطة (ثوانٍ)')
    ap.add_argument('--one', nargs=5, metavar=('BACKBONE', 'CONFIG', 'K', 'MODE', 'THREADS'))
    args = ap.parse_args()

    if args.one:
        bb, cfg, K, mode, th = args.one
        row = measure_point(bb, cfg, int(K), mode, int(th))
        print('RESULT ' + json.dumps(row))
        return

    import pandas as pd
    os.makedirs(RESULTS, exist_ok=True)
    out_csv = os.path.join(RESULTS, 'cpu_ram_vs_frames.csv')
    total, avail = system_ram()
    print(f"[RAM] الكلية {total / 2 ** 30:.1f} GB | المتاحة الآن {avail / 2 ** 30:.1f} GB | خيوط: {args.threads}",
          flush=True)
    rows = []
    for spec in args.models:
        bb, cfg = spec.split(':')
        last_ok = None     # آخر نقطة all_at_once ناجحة: (K، الزيادة، الأساس)
        stopped = None     # سبب إيقاف all_at_once لبقية الأعداد الأكبر
        for K in K_LIST:
            for mode in MODES:
                row = None
                if mode == 'all_at_once' and last_ok:
                    k0, inc0, base0 = last_ok
                    need_mb = base0 + inc0 * K / k0 * GROWTH_MARGIN  # توقّع خطي من آخر نقطة ناجحة
                    avail_mb = system_ram()[1] / MB
                    if stopped or need_mb > SAFETY_FRACTION * avail_mb:
                        stopped = stopped or 'predicted'
                        row = dict(backbone=bb, config=cfg, frames=K, mode=mode, predicted_need_mb=need_mb,
                                   avail_ram_mb=avail_mb,
                                   status=f'لم يُشغَّل: يحتاج ~{need_mb / 1024:.1f} GB (المتاح {avail_mb / 1024:.1f} GB)')
                elif mode == 'all_at_once' and stopped:
                    row = dict(backbone=bb, config=cfg, frames=K, mode=mode, status=f'لم يُشغَّل ({stopped})')
                if row is None:
                    row = run_point(bb, cfg, K, mode, args.threads, args.timeout)
                    if mode == 'all_at_once':
                        if row.get('status') == 'ok':
                            last_ok = (K, row['peak_ram_increase_mb'], row['baseline_mb'])
                        else:
                            stopped = f"بعد {row.get('status')} عند {K} إطاراً"
                            last_ok = None
                rows.append(row)
                pd.DataFrame(rows).to_csv(out_csv, index=False)  # حفظ تدريجي
                extra = (f"  زيادة RAM = {row['peak_ram_increase_mb']:.0f} MB  (الزمن {row.get('time_s', float('nan')):.1f} s)"
                         if 'peak_ram_increase_mb' in row else '')
                print(f"  {bb}/{cfg}  K={K:5d} {mode:12s} → {row.get('status')}{extra}", flush=True)
    df = pd.DataFrame(rows)
    write_summary(df, os.path.join(RESULTS, 'cpu_ram_vs_frames.md'), args.threads, total / MB)
    print(f"[✓] {out_csv}")


if __name__ == '__main__':
    main()
