
### SUMME  (mean ± std عبر الـ seeds؛ F1 بالنسبة المئوية)

| Backbone | Config | F1 strict | F1 standard | Norm. strict | τ (ann) | ρ (ann) | p vs random | p vs pure (F1) | p vs proposed (τ) |
|---|---|---|---|---|---|---|---|---|---|
| mobilevitv2_050 | pure | 49.67 ± 0.73 | 40.06 ± 2.15 | 41.5% | 0.0138 ± 0.0269 | 0.0206 ± 0.0354 | 0.010 | nan | 0.751 |
| mobilevitv2_050 | surgery_only | 45.98 ± 1.99 | 41.69 ± 0.37 | 27.3% | 0.0346 ± 0.0349 | 0.0464 ± 0.0470 | 0.085 | 0.475 | 0.791 |
| mobilevitv2_050 | full_temporal | 46.72 ± 2.66 | 41.64 ± 1.60 | 30.1% | -0.0141 ± 0.0146 | -0.0166 ± 0.0227 | 0.090 | 0.833 | 0.173 |
| mobilevitv2_050 | proposed | 47.42 ± 2.16 | 43.82 ± 1.82 | 32.8% | 0.0420 ± 0.0130 | 0.0582 ± 0.0181 | 0.063 | 0.833 | nan |
| mobilevit_xxs | pure | 46.80 ± 2.09 | 38.34 ± 0.24 | 30.4% | 0.0221 ± 0.0145 | 0.0307 ± 0.0203 | 0.067 | nan | 0.411 |
| mobilevit_xxs | surgery_only | 47.46 ± 4.38 | 42.07 ± 2.64 | 33.0% | 0.0440 ± 0.0077 | 0.0613 ± 0.0109 | 0.026 | 0.916 | 0.791 |
| mobilevit_xxs | full_temporal | 47.25 ± 5.69 | 42.96 ± 1.96 | 32.2% | 0.0093 ± 0.0041 | 0.0127 ± 0.0065 | 0.080 | 0.916 | 0.442 |
| mobilevit_xxs | proposed | 44.44 ± 6.17 | 43.98 ± 1.25 | 21.3% | 0.0030 ± 0.0246 | 0.0058 ± 0.0329 | 0.230 | 0.560 | 0.458 |
| mobilenetv3_small_100 | pure | 51.87 ± 2.02 | 37.63 ± 1.14 | 50.0% | 0.0438 ± 0.0060 | 0.0608 ± 0.0124 | 0.002 | nan | 0.833 |
| mobilenetv3_small_100 | surgery_only | 46.38 ± 2.11 | 40.16 ± 1.80 | 28.8% | 0.0475 ± 0.0264 | 0.0658 ± 0.0381 | 0.101 | 0.220 | 0.791 |
| mobilenetv3_small_100 | full_temporal | 45.94 ± 4.57 | 40.57 ± 2.20 | 27.1% | 0.0421 ± 0.0026 | 0.0607 ± 0.0046 | 0.059 | 0.030 | 0.958 |
| mobilenetv3_small_100 | proposed | 46.62 ± 4.47 | 42.67 ± 2.44 | 29.7% | 0.0085 ± 0.0144 | 0.0137 ± 0.0191 | 0.052 | 0.101 | 0.353 |
| googlenet | full_temporal | 43.14 ± 1.86 | 42.35 ± 1.20 | 16.3% | 0.0031 ± 0.0118 | 0.0039 ± 0.0180 | 0.173 | nan | 0.560 |
| resnet50 | full_temporal | 45.61 ± 4.13 | 40.00 ± 0.60 | 25.9% | -0.0364 ± 0.0217 | -0.0482 ± 0.0303 | 0.075 | nan | 0.127 |
| mobilenetv3_large_100 | full_temporal | 45.51 ± 4.38 | 39.78 ± 3.35 | 25.4% | -0.0117 ± 0.0512 | -0.0158 ± 0.0664 | 0.096 | nan | 0.411 |
| vit_base_patch16_224 | full_temporal | 50.57 ± 3.03 | 40.20 ± 0.47 | 45.0% | 0.0461 ± 0.0136 | 0.0632 ± 0.0154 | 0.003 | nan | 0.853 |
| — | Random | 38.92 | 36.26 | 0% | 0 | 0 | | | |
| — | Human (LOO) | 64.82 | 63.96 | 100% | — | | | | |

### SUMME — اختبار التكافؤ مقابل النموذج المقترح v2-050 (الفرق = المقارَن − المقترح؛ فترة ثقة 95% bootstrap؛ TOST)

| Compared | Params × proposed | Metric | Δ mean | 95% CI | margin | TOST p | Verdict |
|---|---|---|---|---|---|---|---|
| mobilevitv2_050/full_temporal | 1.91× | tau_ann | -0.056 | [-0.137, +0.025] | ±0.02 | 0.801 | inconclusive |
| mobilevitv2_050/full_temporal | 1.91× | f1_strict | -0.703 | [-9.091, +7.121] | ±1.5 | 0.426 | inconclusive |
| mobilevitv2_050/pure | 1.72× | tau_ann | -0.028 | [-0.120, +0.064] | ±0.02 | 0.567 | inconclusive |
| mobilevitv2_050/pure | 1.72× | f1_strict | +2.241 | [-5.852, +10.734] | ±1.5 | 0.567 | inconclusive |
| mobilevit_xxs/proposed | 0.95× | tau_ann | -0.039 | [-0.106, +0.026] | ±0.02 | 0.707 | inconclusive |
| mobilevit_xxs/proposed | 0.95× | f1_strict | -2.987 | [-10.488, +4.617] | ±1.5 | 0.644 | inconclusive |
| mobilenetv3_small_100/proposed | 0.49× | tau_ann | -0.034 | [-0.127, +0.063] | ±0.02 | 0.607 | inconclusive |
| mobilenetv3_small_100/proposed | 0.49× | f1_strict | -0.807 | [-9.156, +7.708] | ±1.5 | 0.438 | inconclusive |
| mobilenetv3_large_100/full_temporal | 6.69× | tau_ann | -0.054 | [-0.145, +0.038] | ±0.02 | 0.762 | inconclusive |
| mobilenetv3_large_100/full_temporal | 6.69× | f1_strict | -1.919 | [-10.183, +6.334] | ±1.5 | 0.538 | inconclusive |
| googlenet/full_temporal | 8.78× | tau_ann | -0.039 | [-0.133, +0.053] | ±0.02 | 0.650 | inconclusive |
| googlenet/full_temporal | 8.78× | f1_strict | -4.289 | [-11.824, +3.596] | ±1.5 | 0.749 | inconclusive |
| resnet50/full_temporal | 36.00× | tau_ann | -0.078 | [-0.158, -0.002] | ±0.02 | 0.919 | inconclusive |
| resnet50/full_temporal | 36.00× | f1_strict | -1.811 | [-8.692, +5.799] | ±1.5 | 0.532 | inconclusive |
| vit_base_patch16_224/full_temporal | 130.24× | tau_ann | +0.004 | [-0.083, +0.087] | ±0.02 | 0.363 | inconclusive |
| vit_base_patch16_224/full_temporal | 130.24× | f1_strict | +3.144 | [-4.911, +11.512] | ±1.5 | 0.648 | inconclusive |

### TVSUM  (mean ± std عبر الـ seeds؛ F1 بالنسبة المئوية)

| Backbone | Config | F1 strict | F1 standard | Norm. strict | τ (ann) | ρ (ann) | p vs random | p vs pure (F1) | p vs proposed (τ) |
|---|---|---|---|---|---|---|---|---|---|
| mobilevitv2_050 | pure | 20.93 ± 0.84 | 46.24 ± 0.67 | 35.2% | 0.1038 ± 0.0095 | 0.1361 ± 0.0127 | 0.001 | nan | 0.058 |
| mobilevitv2_050 | surgery_only | 22.19 ± 0.39 | 47.10 ± 0.52 | 49.9% | 0.1320 ± 0.0071 | 0.1726 ± 0.0094 | 0.000 | 0.328 | 0.472 |
| mobilevitv2_050 | full_temporal | 22.57 ± 0.61 | 46.45 ± 0.52 | 54.3% | 0.1418 ± 0.0099 | 0.1852 ± 0.0129 | 0.000 | 0.033 | 0.287 |
| mobilevitv2_050 | proposed | 22.32 ± 0.38 | 45.79 ± 0.21 | 51.5% | 0.1289 ± 0.0055 | 0.1686 ± 0.0075 | 0.000 | 0.291 | nan |
| mobilevit_xxs | pure | 20.39 ± 0.78 | 47.27 ± 0.65 | 28.9% | 0.1165 ± 0.0072 | 0.1530 ± 0.0094 | 0.021 | nan | 0.097 |
| mobilevit_xxs | surgery_only | 21.19 ± 0.83 | 46.51 ± 0.38 | 38.2% | 0.1154 ± 0.0090 | 0.1515 ± 0.0119 | 0.002 | 0.121 | 0.160 |
| mobilevit_xxs | full_temporal | 21.19 ± 0.51 | 45.03 ± 0.59 | 38.3% | 0.1202 ± 0.0074 | 0.1578 ± 0.0094 | 0.002 | 0.215 | 0.309 |
| mobilevit_xxs | proposed | 20.55 ± 0.95 | 44.70 ± 0.50 | 30.8% | 0.1231 ± 0.0112 | 0.1617 ± 0.0149 | 0.035 | 0.449 | 0.399 |
| mobilenetv3_small_100 | pure | 21.79 ± 0.87 | 48.15 ± 0.39 | 45.3% | 0.1288 ± 0.0057 | 0.1687 ± 0.0076 | 0.002 | nan | 0.789 |
| mobilenetv3_small_100 | surgery_only | 19.57 ± 0.55 | 46.78 ± 0.28 | 19.4% | 0.0977 ± 0.0047 | 0.1287 ± 0.0063 | 0.086 | 0.013 | 0.010 |
| mobilenetv3_small_100 | full_temporal | 20.42 ± 1.47 | 47.39 ± 0.35 | 29.2% | 0.1274 ± 0.0074 | 0.1669 ± 0.0093 | 0.018 | 0.036 | 0.478 |
| mobilenetv3_small_100 | proposed | 20.38 ± 0.20 | 45.72 ± 0.36 | 28.8% | 0.1047 ± 0.0037 | 0.1377 ± 0.0049 | 0.045 | 0.169 | 0.013 |
| googlenet | full_temporal | 22.82 ± 0.91 | 48.97 ± 0.06 | 57.3% | 0.1528 ± 0.0033 | 0.1999 ± 0.0043 | 0.000 | nan | 0.013 |
| resnet50 | full_temporal | 22.62 ± 0.96 | 47.59 ± 0.61 | 54.9% | 0.1494 ± 0.0084 | 0.1957 ± 0.0114 | 0.000 | nan | 0.022 |
| mobilenetv3_large_100 | full_temporal | 21.69 ± 1.42 | 47.88 ± 0.24 | 44.1% | 0.1432 ± 0.0053 | 0.1873 ± 0.0064 | 0.002 | nan | 0.245 |
| vit_base_patch16_224 | full_temporal | 22.82 ± 0.90 | 49.21 ± 0.42 | 57.2% | 0.1622 ± 0.0043 | 0.2123 ± 0.0057 | 0.000 | nan | 0.001 |
| — | Random | 17.91 | 42.91 | 0% | 0 | 0 | | | |
| — | Human (LOO) | 26.48 | 44.60 | 100% | 0.1790 | | | | |

### TVSUM — اختبار التكافؤ مقابل النموذج المقترح v2-050 (الفرق = المقارَن − المقترح؛ فترة ثقة 95% bootstrap؛ TOST)

| Compared | Params × proposed | Metric | Δ mean | 95% CI | margin | TOST p | Verdict |
|---|---|---|---|---|---|---|---|
| mobilevitv2_050/full_temporal | 1.91× | tau_ann | +0.013 | [-0.010, +0.037] | ±0.02 | 0.279 | inconclusive |
| mobilevitv2_050/full_temporal | 1.91× | f1_strict | +0.244 | [-1.927, +2.427] | ±1.5 | 0.133 | inconclusive |
| mobilevitv2_050/pure | 1.72× | tau_ann | -0.025 | [-0.052, +0.000] | ±0.02 | 0.648 | inconclusive |
| mobilevitv2_050/pure | 1.72× | f1_strict | -1.394 | [-3.379, +0.480] | ±1.5 | 0.458 | inconclusive |
| mobilevit_xxs/proposed | 0.95× | tau_ann | -0.006 | [-0.026, +0.016] | ±0.02 | 0.097 | inconclusive |
| mobilevit_xxs/proposed | 0.95× | f1_strict | -1.777 | [-3.983, +0.353] | ±1.5 | 0.597 | inconclusive |
| mobilenetv3_small_100/proposed | 0.49× | tau_ann | -0.024 | [-0.046, -0.002] | ±0.02 | 0.641 | inconclusive |
| mobilenetv3_small_100/proposed | 0.49× | f1_strict | -1.941 | [-4.018, +0.000] | ±1.5 | 0.662 | inconclusive |
| mobilenetv3_large_100/full_temporal | 6.69× | tau_ann | +0.014 | [-0.004, +0.034] | ±0.02 | 0.280 | inconclusive |
| mobilenetv3_large_100/full_temporal | 6.69× | f1_strict | -0.636 | [-2.514, +1.280] | ±1.5 | 0.192 | inconclusive |
| googlenet/full_temporal | 8.78× | tau_ann | +0.024 | [+0.007, +0.042] | ±0.02 | 0.670 | inconclusive |
| googlenet/full_temporal | 8.78× | f1_strict | +0.497 | [-1.156, +2.367] | ±1.5 | 0.138 | inconclusive |
| resnet50/full_temporal | 36.00× | tau_ann | +0.020 | [+0.005, +0.037] | ±0.02 | 0.523 | inconclusive |
| resnet50/full_temporal | 36.00× | f1_strict | +0.297 | [-1.503, +2.102] | ±1.5 | 0.101 | inconclusive |
| vit_base_patch16_224/full_temporal | 130.24× | tau_ann | +0.033 | [+0.014, +0.054] | ±0.02 | 0.900 | inconclusive |
| vit_base_patch16_224/full_temporal | 130.24× | f1_strict | +0.492 | [-1.458, +2.576] | ±1.5 | 0.171 | inconclusive |

### اختيار النموذج الأساسي (قاعدة مُثبَّتة مسبقاً: عدم الدونية في τ على TVSum، الهامش 0.02)

| Candidate | Params | Seeds | τ | Δ vs v2-050 | 95% CI | Non-inferior |
|---|---|---|---|---|---|---|
| mobilenetv3_small_100/proposed | 322,809 | 3 | 0.1047 | -0.0242 | [-0.0464, -0.0024] | no |
| mobilevit_xxs/proposed | 624,657 | 3 | 0.1231 | -0.0058 | [-0.0261, +0.0161] | no |
| mobilevitv2_050/proposed | 660,151 | 3 | 0.1289 | +0.0000 | [+0.0000, +0.0000] | yes |

**النموذج الأساسي: mobilevitv2_050/proposed** (نهائي)

### Efficiency (الزمن = وسيط ms لكل فيديو من 30 إطاراً على الـ GPU)

dataset              backbone        config   params  size_fp16_mb  gflops_per_video  latency_ms  peak_mem_mb  realtime_factor_model  decode_ms  realtime_factor_end_to_end
  summe             googlenet full_temporal  5794657         11.05             89.85       13.69       214.67               11255.20     868.65                      158.11
  summe mobilenetv3_large_100 full_temporal  4413169          8.42             12.93        7.97       194.56               18123.23     868.65                      158.93
  summe mobilenetv3_small_100 full_temporal  1712609          3.27              3.34        7.77       110.19               18793.78     868.65                      158.81
  summe mobilenetv3_small_100      proposed   322809          0.62              2.12        5.48        92.59               27506.74     868.65                      159.34
  summe mobilenetv3_small_100          pure  1587681          3.03              3.33        6.03       107.72               23160.64     868.65                      159.32
  summe mobilenetv3_small_100  surgery_only   197881          0.38              2.11        4.47        91.66               30873.47     868.65                      159.24
  summe         mobilevit_xxs full_temporal  1100721          2.10             18.08       11.77       147.33               12521.27     868.65                      158.46
  summe         mobilevit_xxs      proposed   624657          1.19             16.44       13.29       144.98               11300.73     868.65                      157.94
  summe         mobilevit_xxs          pure   975793          1.86             18.07       10.21       146.07               14762.27     868.65                      158.62
  summe         mobilevit_xxs  surgery_only   499729          0.95             16.43        8.22       143.72               18946.34     868.65                      158.92
  summe       mobilevitv2_050 full_temporal  1259194          2.40             21.71       22.96       193.20                6463.64     868.65                      156.83
  summe       mobilevitv2_050      proposed   660151          1.26             18.96       10.01       179.75               15270.10     868.65                      158.67
  summe       mobilevitv2_050          pure  1134266          2.16             21.70       12.68       192.27                9614.41     868.65                      158.43
  summe       mobilevitv2_050  surgery_only   535223          1.02             18.95       10.11       178.49               14124.80     868.65                      158.61
  summe              resnet50 full_temporal 23768321         45.33            245.24       24.59       303.63                6326.71     868.65                      156.52
  summe  vit_base_patch16_224 full_temporal 85977025        163.99           1053.79       59.27       701.33                2602.75     868.65                      151.66
  tvsum             googlenet full_temporal  5794657         11.05             89.85       13.75       214.19               14327.99     346.13                      555.47
  tvsum mobilenetv3_large_100 full_temporal  4413169          8.42             12.93        7.98       194.56               23987.22     346.13                      563.44
  tvsum mobilenetv3_small_100 full_temporal  1712609          3.27              3.34        7.16       110.19               27292.64     346.13                      565.89
  tvsum mobilenetv3_small_100      proposed   322809          0.62              2.12        6.00        92.59               33309.06     346.13                      565.30
  tvsum mobilenetv3_small_100          pure  1587681          3.03              3.33        5.81       107.98               34572.64     346.13                      568.28
  tvsum mobilenetv3_small_100  surgery_only   197881          0.38              2.11        8.36        91.66               23405.19     346.13                      563.98
  tvsum         mobilevit_xxs full_temporal  1100721          2.10             18.08       10.79       147.16               18482.53     346.13                      560.32
  tvsum         mobilevit_xxs      proposed   624657          1.19             16.44        8.61       144.81               23238.49     346.13                      563.52
  tvsum         mobilevit_xxs          pure   975793          1.86             18.07        9.51       146.07               20735.03     346.13                      562.17
  tvsum         mobilevit_xxs  surgery_only   499729          0.95             16.43        8.19       143.72               24155.88     346.13                      564.11
  tvsum       mobilevitv2_050 full_temporal  1259194          2.40             21.71       12.02       193.20               16264.81     346.13                      558.22
  tvsum       mobilevitv2_050      proposed   660151          1.26             18.96       10.00       179.42               19866.05     346.13                      561.34
  tvsum       mobilevitv2_050          pure  1134266          2.16             21.70       11.55       192.27               17174.07     346.13                      558.96
  tvsum       mobilevitv2_050  surgery_only   535223          1.02             18.95        9.56       178.49               20772.60     346.13                      562.06
  tvsum              resnet50 full_temporal 23768321         45.33            245.24       24.41       303.89                8168.55     346.13                      539.51
  tvsum  vit_base_patch16_224 full_temporal 85977025        163.99           1053.79       59.14       701.43                3364.54     346.13                      495.34
[0m