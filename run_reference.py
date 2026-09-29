"""
المقارنة العادلة (الخيار أ) + اختبار تعميم الجراحة خارج MobileViT:

  1) MobileNetV3-Small بتصميم 2×2 الكامل (pure، surgery_only، full_temporal، proposed)
     — نفس مبدأ الجراحة (حذف مرحلة 1/32) ونفس الرأس الزمني: هل تعمل الجراحة على CNN خفيف أيضاً؟
  2) backbones مرجعية أثقل، كاملة، بنفس الرأس الزمني (full_temporal فقط):
     GoogLeNet (أساس ميزات معظم طرق Table 1)، ResNet-50، MobileNetV3-Large، ViT-Base/16.

كلها بنفس وصفة التدريب، ونفس الـ folds، ونفس البروتوكول.

⚠️ شغّله بعد انتهاء run_all.py (GPU واحد؛ التشغيل المتزامن يبطئ الاثنين ويفسد قياس الزمن).
في PyCharm: زر الفأرة الأيمن ← Run 'run_reference'. قابل للاستئناف مثل run_all.py.

اختبار سريع أولاً (دقائق): أضف --smoke في Run Configuration → Parameters.
"""
import os
import sys
import argparse
import subprocess
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
ENV = dict(os.environ, PYTHONIOENCODING='utf-8')

SEEDS = [42, 43, 44]
DATASETS = ['summe', 'tvsum']
JOBS = ([('mobilenetv3_small_100', c) for c in ['proposed', 'pure', 'surgery_only', 'full_temporal']] +
        [(bb, 'full_temporal') for bb in ['googlenet', 'resnet50', 'mobilenetv3_large_100', 'vit_base_patch16_224']])


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
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='fold واحد وepoch واحد لكل نموذج على SumMe')
    args = ap.parse_args()

    if args.smoke:
        for bb, cfg in JOBS:
            common = ['--dataset', 'summe', '--backbone', bb, '--config', cfg, '--seed', 42, '--tag', 'smoke']
            run('train_cv.py', common + ['--folds', 1, '--epochs', 1])
            run('evaluate_cv.py', common)
        print("\n[✓] الاختبار السريع نجح لكل النماذج. شغّل run_reference.py بلا --smoke.")
        return

    total = len(SEEDS) * len(DATASETS) * len(JOBS)
    n = 0
    for seed in SEEDS:
        for ds in DATASETS:
            for bb, cfg in JOBS:
                n += 1
                print(f"\n########## [{n}/{total}] seed={seed} | {ds} | {bb} | {cfg} ##########", flush=True)
                if evaluated(ds, bb, cfg, seed):
                    print("[skip] مكتمل مسبقاً")
                    continue
                a = ['--dataset', ds, '--backbone', bb, '--config', cfg, '--seed', seed]
                run('train_cv.py', a)
                run('evaluate_cv.py', a)
        run('summarize_results.py', [], out_file=os.path.join(RESULTS, f"summary_with_reference_after_seed_{seed}.md"))

    run('summarize_results.py', [], out_file=os.path.join(RESULTS, 'summary.md'))
    print(f"\n[✓] انتهى. الجداول الكاملة (كل النماذج) في: {os.path.join(RESULTS, 'summary.md')}")


if __name__ == '__main__':
    main()
