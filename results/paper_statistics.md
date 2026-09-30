### TVSUM: paired effects on Kendall's tau (Table 3)

| Backbone | Effect | Difference [95% CI] | Wilcoxon p |
|---|---|---|---|
| mobilevitv2_050 | surgery without head (surgery only - pure) | +0.028 [+0.005, +0.053] | 0.024 |
| mobilevitv2_050 | head without surgery (full + head - pure) | +0.038 [+0.026, +0.051] | <0.001 |
| mobilevitv2_050 | head after surgery (surgery + head - surgery only) | -0.003 [-0.011, +0.005] | 0.472 |
| mobilevitv2_050 | surgery with head (surgery + head - full + head) | -0.013 [-0.037, +0.010] | 0.287 |
| mobilevit_xxs | surgery without head (surgery only - pure) | -0.001 [-0.023, +0.020] | 0.826 |
| mobilevit_xxs | head without surgery (full + head - pure) | +0.004 [-0.010, +0.018] | 0.774 |
| mobilevit_xxs | head after surgery (surgery + head - surgery only) | +0.008 [-0.006, +0.022] | 0.226 |
| mobilevit_xxs | surgery with head (surgery + head - full + head) | +0.003 [-0.019, +0.024] | 0.455 |
| mobilenetv3_small_100 | surgery without head (surgery only - pure) | -0.031 [-0.049, -0.013] | 0.002 |
| mobilenetv3_small_100 | head without surgery (full + head - pure) | -0.001 [-0.013, +0.009] | 0.992 |
| mobilenetv3_small_100 | head after surgery (surgery + head - surgery only) | +0.007 [-0.012, +0.023] | 0.198 |
| mobilenetv3_small_100 | surgery with head (surgery + head - full + head) | -0.023 [-0.040, -0.006] | 0.033 |

### TVSUM: each configuration against the proposed model (compared - proposed, tau)

| Compared | Pre-specified | Difference [95% CI] | Wilcoxon p | TOST p (+-0.02) |
|---|---|---|---|---|
| googlenet/full_temporal | yes | +0.024 [+0.007, +0.042] | 0.013 | 0.670 |
| mobilenetv3_large_100/full_temporal | yes | +0.014 [-0.004, +0.034] | 0.245 | 0.280 |
| mobilenetv3_small_100/proposed | yes | -0.024 [-0.046, -0.002] | 0.013 | 0.641 |
| mobilevit_xxs/proposed | yes | -0.006 [-0.026, +0.016] | 0.399 | 0.097 |
| mobilevitv2_050/full_temporal | yes | +0.013 [-0.010, +0.037] | 0.287 | 0.279 |
| mobilevitv2_050/pure | yes | -0.025 [-0.052, 0.000] | 0.058 | 0.648 |
| resnet50/full_temporal | yes | +0.020 [+0.005, +0.037] | 0.022 | 0.523 |
| vit_base_patch16_224/full_temporal | yes | +0.033 [+0.014, +0.054] | 0.001 | 0.900 |
| mobilenetv3_small_100/full_temporal | no (exploratory) | -0.002 [-0.018, +0.016] | 0.478 | 0.021 |
| mobilenetv3_small_100/pure | no (exploratory) | 0.000 [-0.016, +0.016] | 0.789 | 0.010 |
| mobilenetv3_small_100/surgery_only | no (exploratory) | -0.031 [-0.054, -0.007] | 0.010 | 0.816 |
| mobilevit_xxs/full_temporal | no (exploratory) | -0.009 [-0.032, +0.014] | 0.309 | 0.174 |
| mobilevit_xxs/pure | no (exploratory) | -0.012 [-0.033, +0.007] | 0.097 | 0.232 |
| mobilevit_xxs/surgery_only | no (exploratory) | -0.014 [-0.033, +0.007] | 0.160 | 0.264 |
| mobilevitv2_050/surgery_only | no (exploratory) | +0.003 [-0.005, +0.011] | 0.472 | <0.001 |

### SUMME: paired effects on Kendall's tau

| Backbone | Effect | Difference [95% CI] | Wilcoxon p |
|---|---|---|---|
| mobilevitv2_050 | surgery without head (surgery only - pure) | +0.021 [-0.063, +0.107] | 0.979 |
| mobilevitv2_050 | head without surgery (full + head - pure) | -0.028 [-0.067, +0.012] | 0.252 |
| mobilevitv2_050 | head after surgery (surgery + head - surgery only) | +0.007 [-0.019, +0.035] | 0.791 |
| mobilevitv2_050 | surgery with head (surgery + head - full + head) | +0.056 [-0.025, +0.137] | 0.173 |
| mobilevit_xxs | surgery without head (surgery only - pure) | +0.022 [-0.031, +0.078] | 0.560 |
| mobilevit_xxs | head without surgery (full + head - pure) | -0.013 [-0.068, +0.039] | 0.751 |
| mobilevit_xxs | head after surgery (surgery + head - surgery only) | -0.041 [-0.089, +0.005] | 0.230 |
| mobilevit_xxs | surgery with head (surgery + head - full + head) | -0.006 [-0.044, +0.033] | 0.751 |
| mobilenetv3_small_100 | surgery without head (surgery only - pure) | +0.004 [-0.050, +0.059] | 1.000 |
| mobilenetv3_small_100 | head without surgery (full + head - pure) | -0.002 [-0.042, +0.034] | 0.458 |
| mobilenetv3_small_100 | head after surgery (surgery + head - surgery only) | -0.039 [-0.085, +0.010] | 0.067 |
| mobilenetv3_small_100 | surgery with head (surgery + head - full + head) | -0.034 [-0.086, +0.021] | 0.182 |

### SUMME: each configuration against the proposed model (compared - proposed, tau)

| Compared | Pre-specified | Difference [95% CI] | Wilcoxon p | TOST p (+-0.02) |
|---|---|---|---|---|
| googlenet/full_temporal | yes | -0.039 [-0.133, +0.053] | 0.560 | 0.650 |
| mobilenetv3_large_100/full_temporal | yes | -0.054 [-0.145, +0.038] | 0.411 | 0.762 |
| mobilenetv3_small_100/proposed | yes | -0.034 [-0.127, +0.063] | 0.353 | 0.607 |
| mobilevit_xxs/proposed | yes | -0.039 [-0.106, +0.026] | 0.458 | 0.707 |
| mobilevitv2_050/full_temporal | yes | -0.056 [-0.137, +0.025] | 0.173 | 0.801 |
| mobilevitv2_050/pure | yes | -0.028 [-0.120, +0.064] | 0.751 | 0.567 |
| resnet50/full_temporal | yes | -0.078 [-0.158, -0.002] | 0.127 | 0.919 |
| vit_base_patch16_224/full_temporal | yes | +0.004 [-0.083, +0.087] | 0.853 | 0.363 |
| mobilenetv3_small_100/full_temporal | no (exploratory) | 0.000 [-0.087, +0.088] | 0.958 | 0.333 |
| mobilenetv3_small_100/pure | no (exploratory) | +0.002 [-0.099, +0.108] | 0.833 | 0.368 |
| mobilenetv3_small_100/surgery_only | no (exploratory) | +0.006 [-0.097, +0.112] | 0.791 | 0.395 |
| mobilevit_xxs/full_temporal | no (exploratory) | -0.033 [-0.108, +0.042] | 0.442 | 0.625 |
| mobilevit_xxs/pure | no (exploratory) | -0.020 [-0.118, +0.082] | 0.411 | 0.499 |
| mobilevit_xxs/surgery_only | no (exploratory) | +0.002 [-0.086, +0.089] | 0.791 | 0.349 |
| mobilevitv2_050/surgery_only | no (exploratory) | -0.007 [-0.035, +0.019] | 0.791 | 0.189 |
