"""
التشغيل الكامل لكل التجارب (بديل run_all.bat لمن يستخدم PyCharm).
في PyCharm: زر الفأرة الأيمن على هذا الملف ← Run 'run_all'.

- قابل للاستئناف: إذا توقف (إغلاق PyCharm، انقطاع كهرباء...) شغّله مرة أخرى وسيكمل من حيث توقف.
- بعد كل seed يكتب ملخصاً مرحلياً في results/summary_after_seed_XX.md
- أبقِ الحاسوب مستيقظاً (Power settings → Sleep: Never) طوال التشغيل.
"""
import os
import sys
import subprocess
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
ENV = dict(os.environ, PYTHONIOENCODING='utf-8')

SEEDS = [42, 43, 44]
DATASETS = ['summe', 'tvsum']
BACKBONES = ['mobilevitv2_050', 'mobilevit_xxs']
CONFIGS = ['proposed', 'pure', 'surgery_only', 'full_temporal']


MAX_TRIES = 3          # إعادة المحاولة تلقائياً بعد أي انهيار (التدريب يكمل من آخر fold محفوظ)
PAUSE_S = 30


def run(script, args, out_file=None):
    cmd = [sys.executable, '-X', 'faulthandler', script, *[str(a) for a in args]]
    for attempt in range(1, MAX_TRIES + 1):
        print('\n>>> ' + ' '.join(cmd[3:]) + (f'   (محاولة {attempt}/{MAX_TRIES})' if attempt > 1 else ''), flush=True)
        if out_file:
            with open(out_file, 'w', encoding='utf-8') as f:
                code = subprocess.run(cmd, cwd=HERE, env=ENV, stdout=f).returncode
        else:
            code = subprocess.run(cmd, cwd=HERE, env=ENV).returncode
        if code == 0:
            return
        msg = f"{' '.join(cmd[3:])} | exit code {code} (hex {code & 0xFFFFFFFF:#x}) | attempt {attempt}"
        print(f"\n[!] {msg}", flush=True)
        os.makedirs(RESULTS, exist_ok=True)
        with open(os.path.join(RESULTS, 'crash_log.txt'), 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
        if attempt < MAX_TRIES:
            print(f"[i] إعادة المحاولة بعد {PAUSE_S} ثانية...", flush=True)
            time.sleep(PAUSE_S)
    sys.exit(f"\n[!] توقف بعد {MAX_TRIES} محاولات فاشلة في {script}. أرسل آخر المخرجات وملف results/crash_log.txt.")


def evaluated(ds, bb, cfg, seed):
    p = os.path.join(RESULTS, f"per_video_{ds}_{bb}_{cfg}_seed{seed}.csv")
    return os.path.exists(p) and pd.read_csv(p).fold.nunique() == 5


def main():
    os.makedirs(RESULTS, exist_ok=True)

    # الخطوة 1: cache + العشوائي + السقف البشري + زمن فك الترميز (مرة واحدة لكل مجموعة)
    for ds in DATASETS:
        p = os.path.join(RESULTS, f"baselines_{ds}.csv")
        if not (os.path.exists(p) and 'decode_ms' in pd.read_csv(p).columns):
            run('evaluate_cv.py', ['--dataset', ds, '--baselines_only', '--time_decode'])

    # الخطوة 2: تدريب + تقييم (الـ seed في الحلقة الخارجية: بعد أول دورة يكون لكل إعداد نتيجة)
    total = len(SEEDS) * len(DATASETS) * len(BACKBONES) * len(CONFIGS)
    n = 0
    for seed in SEEDS:
        for ds in DATASETS:
            for bb in BACKBONES:
                for cfg in CONFIGS:
                    n += 1
                    print(f"\n########## [{n}/{total}] seed={seed} | {ds} | {bb} | {cfg} ##########", flush=True)
                    if evaluated(ds, bb, cfg, seed):
                        print("[skip] مكتمل مسبقاً")
                        continue
                    args = ['--dataset', ds, '--backbone', bb, '--config', cfg, '--seed', seed]
                    run('train_cv.py', args)
                    run('evaluate_cv.py', args)
        run('summarize_results.py', [], out_file=os.path.join(RESULTS, f"summary_after_seed_{seed}.md"))

    # الخطوة 3: الجداول النهائية
    run('summarize_results.py', [], out_file=os.path.join(RESULTS, 'summary.md'))
    print(f"\n[✓] انتهى كل شيء. الجداول النهائية في: {os.path.join(RESULTS, 'summary.md')}")


if __name__ == '__main__':
    main()
