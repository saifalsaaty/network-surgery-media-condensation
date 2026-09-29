"""
معايرة السقف البشري لـ τ على SumMe بنفس بروتوكولنا (150 نقطة من user_score في ملفات GT).
في PyCharm: زر الفأرة الأيمن ← Run. المخرج: results/summe_human_tau.csv

- loo  : كل مقيّم مقابل متوسط الباقين (المرجع المناسب لنماذجنا، لأنها تُقاس مقابل متوسط المقيّمين)
- pair : متوسط τ بين كل زوج من المقيّمين (يقارن بما تنشره الأدبيات)
النتيجة: loo ≈ 0.269، pair ≈ 0.206
"""
import os
import glob

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import kendalltau

import common as C

rows = []
for f in sorted(glob.glob(os.path.join(C.PATHS['summe']['annotations'], '*.mat'))):
    m = loadmat(f)
    n = len(m['gt_score'].squeeze())
    idx = np.linspace(0, n - 1, C.FINE_FRAMES, dtype=int)
    A = m['user_score'].astype(np.float32)[idx, :].T  # (U, 150)
    U = A.shape[0]
    loo = [kendalltau(A[i], np.delete(A, i, 0).mean(0))[0] for i in range(U)]
    pair = [kendalltau(A[i], A[j])[0] for i in range(U) for j in range(i + 1, U)]
    rows.append(dict(video=os.path.splitext(os.path.basename(f))[0], users=U,
                     human_tau_loo=float(np.nanmean(loo)), human_tau_pairwise=float(np.nanmean(pair))))

df = pd.DataFrame(rows)
os.makedirs(C.RESULTS_DIR, exist_ok=True)
df.to_csv(os.path.join(C.RESULTS_DIR, 'summe_human_tau.csv'), index=False)
print(df.to_string(index=False))
print(f"\nSumMe human τ: leave-one-out = {df.human_tau_loo.mean():.4f} | pairwise = {df.human_tau_pairwise.mean():.4f}")
