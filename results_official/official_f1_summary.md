# F1 بالبروتوكول الرسمي (ملفات h5، change points الرسمية، SumMe = max، TVSum = avg)

mean ± std عبر الـ seeds الثلاثة؛ كل فيديو مُقيَّم من الـ fold الذي كان فيه اختباراً.


## SUMME

| Backbone | Config | F1 official (%) |
|---|---|---|
| mobilevitv2_050 | pure | 42.34 ± 0.82 |
| mobilevitv2_050 | surgery_only | 43.99 ± 2.66 |
| mobilevitv2_050 | full_temporal | 40.02 ± 1.42 |
| mobilevitv2_050 | proposed | 43.34 ± 0.79 |
| mobilevit_xxs | pure | 42.47 ± 2.55 |
| mobilevit_xxs | surgery_only | 40.49 ± 1.30 |
| mobilevit_xxs | full_temporal | 42.85 ± 1.02 |
| mobilevit_xxs | proposed | 39.37 ± 1.54 |
| mobilenetv3_small_100 | pure | 44.20 ± 2.56 |
| mobilenetv3_small_100 | surgery_only | 41.84 ± 3.29 |
| mobilenetv3_small_100 | full_temporal | 44.02 ± 1.82 |
| mobilenetv3_small_100 | proposed | 40.85 ± 3.33 |
| googlenet | full_temporal | 40.37 ± 1.33 |
| resnet50 | full_temporal | 43.67 ± 1.83 |
| mobilenetv3_large_100 | full_temporal | 40.89 ± 0.76 |
| vit_base_patch16_224 | full_temporal | 44.77 ± 2.20 |
| — | Random | 40.31 |
| — | Human (leave-one-out) | 54.33 |

## TVSUM

| Backbone | Config | F1 official (%) |
|---|---|---|
| mobilevitv2_050 | pure | 55.31 ± 0.23 |
| mobilevitv2_050 | surgery_only | 58.43 ± 0.30 |
| mobilevitv2_050 | full_temporal | 55.52 ± 0.82 |
| mobilevitv2_050 | proposed | 56.44 ± 1.08 |
| mobilevit_xxs | pure | 53.45 ± 0.42 |
| mobilevit_xxs | surgery_only | 53.88 ± 0.53 |
| mobilevit_xxs | full_temporal | 52.22 ± 1.64 |
| mobilevit_xxs | proposed | 54.87 ± 0.58 |
| mobilenetv3_small_100 | pure | 55.96 ± 0.29 |
| mobilenetv3_small_100 | surgery_only | 53.84 ± 1.09 |
| mobilenetv3_small_100 | full_temporal | 55.61 ± 1.30 |
| mobilenetv3_small_100 | proposed | 54.24 ± 0.58 |
| googlenet | full_temporal | 56.76 ± 0.38 |
| resnet50 | full_temporal | 55.33 ± 1.40 |
| mobilenetv3_large_100 | full_temporal | 56.80 ± 0.66 |
| vit_base_patch16_224 | full_temporal | 56.39 ± 0.46 |
| — | Random | 54.61 |
| — | Human (leave-one-out) | 53.84 |
