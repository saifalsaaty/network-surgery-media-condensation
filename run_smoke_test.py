"""
اختبار سريع (fold واحد، epoch واحد) للتأكد أن كل شيء يعمل قبل التشغيل الكامل.
في PyCharm: زر الفأرة الأيمن على هذا الملف ← Run 'run_smoke_test'.
الرقم الناتج بلا معنى علمي؛ المطلوب فقط أن ينتهي بلا أخطاء.
"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = dict(os.environ, PYTHONIOENCODING='utf-8')
COMMON = ['--dataset', 'summe', '--backbone', 'mobilevitv2_050', '--config', 'proposed',
          '--seed', '42', '--tag', 'smoke']


def run(script, extra):
    cmd = [sys.executable, '-X', 'faulthandler', script, *extra]
    print('\n>>> ' + ' '.join(cmd[3:]), flush=True)
    code = subprocess.run(cmd, cwd=HERE, env=ENV).returncode
    if code != 0:
        sys.exit(f"\n[!] توقف بسبب خطأ في {script} (exit code {code}, hex {code & 0xFFFFFFFF:#x}). انسخ آخر المخرجات وأرسلها.")


if __name__ == '__main__':
    run('train_cv.py', COMMON + ['--folds', '1', '--epochs', '1'])
    run('evaluate_cv.py', COMMON)
    print("\n[✓] الاختبار السريع نجح. الخطوة التالية: شغّل run_all.py")
