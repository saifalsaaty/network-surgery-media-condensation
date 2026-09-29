# تحليل F1 الرسمي (البند 11؛ وصفي استكشافي، لا يُتخذ عليه قرار)

الفروق بالنقاط المئوية، مزدوجة لكل فيديو بعد أخذ متوسط الـ seeds الثلاثة؛ CI = bootstrap 95% (10000 عيّنة، seed 0)؛ p = Wilcoxon.


## SUMME (25 فيديو)

العشوائي الرسمي = 40.31، البشري (leave-one-out) = 54.33

| Backbone | Config | F1 (%) | Δ عن العشوائي [95% CI] | p | Kendall τ (بروتوكولنا) |
|---|---|---|---|---|---|
| googlenet | full_temporal | 40.37 | +0.06 [-2.81, +3.01] | 0.932 | 0.003 |
| mobilenetv3_large_100 | full_temporal | 40.89 | +0.59 [-3.14, +4.15] | 0.634 | -0.012 |
| mobilenetv3_small_100 | full_temporal | 44.02 | +3.71 [+0.15, +7.16] | 0.052 | 0.042 |
| mobilenetv3_small_100 | proposed | 40.85 | +0.54 [-2.92, +3.87] | 0.560 | 0.008 |
| mobilenetv3_small_100 | pure | 44.20 | +3.89 [+0.91, +6.79] | 0.014 | 0.044 |
| mobilenetv3_small_100 | surgery_only | 41.84 | +1.54 [-2.87, +5.80] | 0.426 | 0.047 |
| mobilevit_xxs | full_temporal | 42.85 | +2.54 [-1.83, +7.28] | 0.275 | 0.009 |
| mobilevit_xxs | proposed | 39.37 | -0.94 [-4.50, +2.34] | 0.710 | 0.003 |
| mobilevit_xxs | pure | 42.47 | +2.16 [-0.55, +4.79] | 0.018 | 0.022 |
| mobilevit_xxs | surgery_only | 40.49 | +0.19 [-3.30, +3.38] | 0.525 | 0.044 |
| mobilevitv2_050 | full_temporal | 40.02 | -0.28 [-2.90, +2.29] | 0.954 | -0.014 |
| mobilevitv2_050 | proposed | 43.34 | +3.04 [+0.11, +6.08] | 0.170 | 0.042 |
| mobilevitv2_050 | pure | 42.34 | +2.03 [-0.65, +4.56] | 0.037 | 0.014 |
| mobilevitv2_050 | surgery_only | 43.99 | +3.69 [+1.36, +6.04] | 0.007 | 0.035 |
| resnet50 | full_temporal | 43.67 | +3.36 [+0.63, +6.14] | 0.026 | -0.036 |
| vit_base_patch16_224 | full_temporal | 44.77 | +4.46 [+1.75, +7.34] | 0.009 | 0.046 |

### mobilevitv2_050 proposed مقابل البقية

| مقابل | Δ F1 [95% CI] | p |
|---|---|---|
| googlenet full_temporal | +2.97 [-0.35, +6.21] | 0.101 |
| mobilenetv3_large_100 full_temporal | +2.45 [-1.20, +6.03] | 0.265 |
| mobilenetv3_small_100 full_temporal | -0.67 [-3.99, +2.74] | 0.508 |
| mobilenetv3_small_100 proposed | +2.50 [-2.21, +7.68] | 0.474 |
| mobilenetv3_small_100 pure | -0.86 [-4.27, +2.83] | 0.587 |
| mobilenetv3_small_100 surgery_only | +1.50 [-3.88, +6.87] | 0.672 |
| mobilevit_xxs full_temporal | +0.49 [-5.17, +5.31] | 0.549 |
| mobilevit_xxs proposed | +3.98 [-0.40, +8.73] | 0.116 |
| mobilevit_xxs pure | +0.88 [-3.10, +4.52] | 0.511 |
| mobilevit_xxs surgery_only | +2.85 [-1.59, +7.77] | 0.381 |
| mobilevitv2_050 full_temporal | +3.32 [-0.53, +7.01] | 0.081 |
| mobilevitv2_050 pure | +1.01 [-1.71, +3.79] | 0.543 |
| mobilevitv2_050 surgery_only | -0.65 [-2.78, +1.47] | 0.681 |
| resnet50 full_temporal | -0.33 [-4.29, +3.49] | 0.927 |
| vit_base_patch16_224 full_temporal | -1.43 [-4.25, +1.38] | 0.274 |

**اتفاق F1 الرسمي مع τ في ترتيب الإعدادات (16):** Spearman = 0.48 (p = 0.058)

**مدى F1 بين كل الإعدادات:** 39.37 – 44.77 (5.40 نقطة)، مقابل فرق البشري عن العشوائي +14.02 نقطة.


## TVSUM (50 فيديو)

العشوائي الرسمي = 54.61، البشري (leave-one-out) = 53.84

| Backbone | Config | F1 (%) | Δ عن العشوائي [95% CI] | p | Kendall τ (بروتوكولنا) |
|---|---|---|---|---|---|
| googlenet | full_temporal | 56.76 | +2.15 [+0.69, +3.53] | 0.005 | 0.153 |
| mobilenetv3_large_100 | full_temporal | 56.80 | +2.19 [+0.76, +3.64] | 0.006 | 0.143 |
| mobilenetv3_small_100 | full_temporal | 55.61 | +0.99 [-0.39, +2.42] | 0.241 | 0.127 |
| mobilenetv3_small_100 | proposed | 54.24 | -0.37 [-1.68, +0.99] | 0.625 | 0.105 |
| mobilenetv3_small_100 | pure | 55.96 | +1.34 [-0.37, +3.03] | 0.084 | 0.129 |
| mobilenetv3_small_100 | surgery_only | 53.84 | -0.77 [-2.55, +1.00] | 0.566 | 0.098 |
| mobilevit_xxs | full_temporal | 52.22 | -2.39 [-4.13, -0.79] | 0.017 | 0.120 |
| mobilevit_xxs | proposed | 54.87 | +0.26 [-1.31, +1.78] | 0.415 | 0.123 |
| mobilevit_xxs | pure | 53.45 | -1.17 [-2.67, +0.27] | 0.274 | 0.117 |
| mobilevit_xxs | surgery_only | 53.88 | -0.73 [-2.35, +0.83] | 0.759 | 0.115 |
| mobilevitv2_050 | full_temporal | 55.52 | +0.90 [-0.56, +2.29] | 0.099 | 0.142 |
| mobilevitv2_050 | proposed | 56.44 | +1.82 [+0.25, +3.28] | 0.002 | 0.129 |
| mobilevitv2_050 | pure | 55.31 | +0.70 [-0.59, +1.99] | 0.300 | 0.104 |
| mobilevitv2_050 | surgery_only | 58.43 | +3.82 [+2.74, +4.89] | 0.000 | 0.132 |
| resnet50 | full_temporal | 55.33 | +0.72 [-0.55, +1.95] | 0.172 | 0.149 |
| vit_base_patch16_224 | full_temporal | 56.39 | +1.78 [+0.18, +3.31] | 0.006 | 0.162 |

### mobilevitv2_050 proposed مقابل البقية

| مقابل | Δ F1 [95% CI] | p |
|---|---|---|
| googlenet full_temporal | -0.32 [-1.97, +1.10] | 0.374 |
| mobilenetv3_large_100 full_temporal | -0.37 [-2.22, +1.27] | 0.988 |
| mobilenetv3_small_100 full_temporal | +0.83 [-1.02, +2.45] | 0.065 |
| mobilenetv3_small_100 proposed | +2.20 [+0.56, +3.81] | 0.004 |
| mobilenetv3_small_100 pure | +0.48 [-1.21, +2.10] | 0.454 |
| mobilenetv3_small_100 surgery_only | +2.60 [+0.79, +4.37] | 0.005 |
| mobilevit_xxs full_temporal | +4.22 [+2.74, +5.63] | 0.000 |
| mobilevit_xxs proposed | +1.56 [-0.08, +3.12] | 0.026 |
| mobilevit_xxs pure | +2.99 [+1.35, +4.63] | 0.000 |
| mobilevit_xxs surgery_only | +2.56 [+0.87, +4.24] | 0.003 |
| mobilevitv2_050 full_temporal | +0.92 [-0.30, +2.19] | 0.200 |
| mobilevitv2_050 pure | +1.12 [-0.17, +2.36] | 0.014 |
| mobilevitv2_050 surgery_only | -1.99 [-3.21, -0.86] | 0.003 |
| resnet50 full_temporal | +1.10 [-0.29, +2.46] | 0.041 |
| vit_base_patch16_224 full_temporal | +0.05 [-1.05, +1.13] | 0.824 |

**اتفاق F1 الرسمي مع τ في ترتيب الإعدادات (16):** Spearman = 0.74 (p = 0.001)

**مدى F1 بين كل الإعدادات:** 52.22 – 58.43 (6.21 نقطة)، مقابل فرق البشري عن العشوائي -0.78 نقطة.

