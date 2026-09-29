"""
إعادة قياس الذاكرة مقابل عدد الإطارات فقط (بديل Table 6)، وكل نقطة في عملية منفصلة.
في PyCharm: زر الفأرة الأيمن ← Run 'run_memory_benchmark'. أغلق البرامج الأخرى أثناءه.

- القياس القديم يُحفظ باسم results/memory_vs_frames_single_process_OLD.csv للمقارنة.
- النتيجة الجديدة في results/memory_vs_frames.csv (تُحفظ تدريجياً بعد كل نقطة).
- لا يلمس نتائج السرعة أو الدقة.
"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
code = subprocess.run([sys.executable, os.path.join(HERE, 'benchmark_speed.py'), '--parts', 'memory'],
                      cwd=HERE).returncode
sys.exit(code)
