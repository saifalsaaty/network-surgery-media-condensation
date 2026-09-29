"""
تجربة الإطارات الأكثف (60 إطاراً) — البند 9 في ANALYSIS_PLAN.md.
في PyCharm: زر الفأرة الأيمن ← Run 'run_frames60'. أبقِ الحاسوب مستيقظاً.

ما يفعله بالترتيب:
  1) يبني cache بـ 60 إطاراً (cache_f60) بنسخ لقطات KTS وتعليقات المقيّمين من cache الـ 30 حرفياً.
  2) اختبار سريع: epoch واحد على fold واحد (للتأكد من أن ذاكرة الـ GPU تكفي). نتائجه لا تدخل التحليل.
  3) التدريب والتقييم: mobilevitv2_050 × (proposed، surgery_only) × 3 seeds — TVSum أولاً ثم SumMe.
  4) التحليل المُثبَّت مسبقاً: analyze_frames60.py ← results_f60/frames60_analysis.md

- قابل للاستئناف: إذا توقف لأي سبب، شغّله مرة أخرى وسيكمل من حيث توقف.
- نتائج الـ 30 إطاراً (results، runs، cache) لا تُلمس إطلاقاً.
"""
import os
import sys
import subprocess
import time

import pandas as pd

os.environ['VS_NUM_FRAMES'] = '60'
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results_f60')
ENV = dict(os.environ, PYTHONIOENCODING='utf-8', VS_NUM_FRAMES='60')

SEEDS = [42, 43, 44]
DATASETS = ['tvsum', 'summe']          # TVSum (الأساسي) أولاً
BACKBONE = 'mobilevitv2_050'
CONFIGS = ['proposed', 'surgery_only']

MAX_TRIES = 3
PAUSE_S = 30


def run(script, args, out_file=None):
    cmd = [sys.executable, '-X', 'faulthandler', script, *[str(a) for a in args]]
    for attempt in range(1, MAX_TRIES + 1):
        print('\n>>> [60 إطاراً] ' + ' '.join(cmd[3:]) +
              (f'   (محاولة {attempt}/{MAX_TRIES})' if attempt > 1 else ''), flush=True)
        if out_file:
            with open(out_file, 'w', encoding='utf-8') as f:
                code = subprocess.run(cmd, cwd=HERE, env=ENV, stdout=f).returncode
        else:
            code = subprocess.run(cmd, cwd=HERE, env=ENV).returncode
        if code == 0:
            return
        msg = (f"{' '.join(cmd[3:])} | exit code {code} "
               f"(hex {code & 0xFFFFFFFF:#x}) | attempt {attempt}")
        print(f"\n[!] {msg}", flush=True)
        os.makedirs(RESULTS, exist_ok=True)
        with open(os.path.join(RESULTS, 'crash_log.txt'), 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
        if attempt < MAX_TRIES:
            print(f"[i] إعادة المحاولة بعد {PAUSE_S} ثانية...", flush=True)
            time.sleep(PAUSE_S)
    sys.exit(f"\n[!] توقف بعد {MAX_TRIES} محاولات فاشلة في {script}. أرسل آخر المخرجات وملف results_f60/crash_log.txt.")


def evaluated(ds, cfg, seed):
    p = os.path.join(RESULTS, f"per_video_{ds}_{BACKBONE}_{cfg}_seed{seed}.csv")
    return os.path.exists(p) and pd.read_csv(p).fold.nunique() == 5


def main():
    os.makedirs(RESULTS, exist_ok=True)

    # 1) cache بـ 60 إطاراً + العشوائي/البشري + زمن فك ترميز 60 إطاراً
    for ds in DATASETS:
        p = os.path.join(RESULTS, f"baselines_{ds}.csv")
        if not (os.path.exists(p) and 'decode_ms' in pd.read_csv(p).columns):
            run('evaluate_cv.py', ['--dataset', ds, '--baselines_only', '--time_decode'])

    # 2) اختبار سريع للذاكرة (epoch واحد، fold واحد، tag منفصل)
    smoke = os.path.join(HERE, 'runs_f60', 'tvsum', BACKBONE, 'proposed', 'seed42_smoke', 'fold_1_last.pth')
    if not os.path.exists(smoke):
        run('train_cv.py', ['--dataset', 'tvsum', '--backbone', BACKBONE, '--config', 'proposed',
                            '--seed', 42, '--folds', 1, '--epochs', 1, '--tag', 'smoke'])
        print("\n[✓] الاختبار السريع نجح: ذاكرة الـ GPU تكفي لـ 60 إطاراً.", flush=True)

    # 3) التدريب والتقييم
    total = len(DATASETS) * len(SEEDS) * len(CONFIGS)
    n = 0
    for ds in DATASETS:
        for seed in SEEDS:
            for cfg in CONFIGS:
                n += 1
                print(f"\n########## [60 إطاراً | {n}/{total}] {ds} | seed={seed} | {cfg} ##########", flush=True)
                if evaluated(ds, cfg, seed):
                    print("[skip] مكتمل مسبقاً")
                    continue
                args = ['--dataset', ds, '--backbone', BACKBONE, '--config', cfg, '--seed', seed]
                run('train_cv.py', args)
                run('evaluate_cv.py', args)
        # 4) التحليل بعد اكتمال كل مجموعة بيانات (TVSum يكفي للقرار)
        run('analyze_frames60.py', [], out_file=os.path.join(RESULTS, 'frames60_analysis.md'))

    print(f"\n[✓] انتهت تجربة الـ 60 إطاراً. التحليل في: {os.path.join(RESULTS, 'frames60_analysis.md')}")


if __name__ == '__main__':
    main()
