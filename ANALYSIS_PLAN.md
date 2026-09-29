# Pre-specified Analysis Plan

**Rule:** no item in this plan was changed after the corresponding results were seen. Any analysis added later is
reported in the paper as exploratory, not as a primary result. Items are listed in the order in which they were fixed.

## 1. Main claim under test

A model with far fewer parameters achieves ranking accuracy approximately equivalent to larger models, under unified
training and evaluation conditions.

## 2. Models

| Role | Model |
|---|---|
| **Primary model** | MobileViT-v2-050 + surgery + two-block Conv1d-Transformer — `mobilevitv2_050 / proposed` |
| Lightweight variant | `mobilevit_xxs / proposed` |
| Ablation (2 × 2) | `pure`, `surgery_only`, `full_temporal`, `proposed` for MobileViT-v2-050, MobileViT-XXS and MobileNetV3-Small |
| Reference backbones | GoogLeNet, ResNet-50, MobileNetV3-Large, ViT-B/16 (full backbone, `full_temporal`) |

**Fixed rule:** the name of the primary model does not change with the results. If another model is significantly
better (p < 0.05 in τ on TVSum across the three seeds), this is reported explicitly in the paper.
**(This rule was amended by item 8: the primary model is now chosen by a pre-specified selection rule.)**

## 3. Training

- Checkpoint selection: **last epoch**; test data are never used for any decision.
- Five-fold cross-validation with a fixed split (`random_state = 42`) and three seeds: 42, 43 and 44.
- One training recipe per dataset, applied to all models (values in `HPARAMS` in `common.py`).
- No hyperparameter is changed after test results are seen.

## 4. Data and metrics

| | |
|---|---|
| **Primary dataset** | **TVSum** (50 videos, 20 annotators) |
| **Primary metric** | **Kendall's τ following Otani et al.** (against each annotator, then averaged, at 150 evaluation points) |
| Secondary metrics | Spearman's ρ and F1 strict (shot value = sum of scores, 15% budget), on TVSum |
| Supporting dataset | SumMe (25 videos): reported in full with confidence intervals; no equivalence claim is based on it because of its size |
| Reference points | Random scores and the human agreement (leave-one-out), with the same protocol |
| Additional protocol | F1 standard (shot value = mean of scores), to show the short-shot bias described by Otani et al. |

## 5. Statistical tests

- Unit of analysis: the **video**. Each video is evaluated only by the model of the fold in which it was a test video,
  and the three seeds are averaged per video.
- Paired video-level comparisons: two-sided Wilcoxon signed-rank test.
- **Equivalence:** 95% percentile-bootstrap confidence interval (10,000 resamples) of the difference, and TOST.
  - Margin for τ: **±0.02** (about 11% of the human τ on TVSum, 0.179).
  - Margin for F1 strict: **±1.5 points**.
  - Decision: *equivalent* if the whole interval lies inside the margin; *better/worse* if it lies entirely outside;
    otherwise *inconclusive*.
- Comparisons with the primary model: `full_temporal` (effect of surgery), `pure`, XXS, MobileNetV3-Small and the four
  reference backbones.

## 6. Efficiency

- Parameter counts include the whole backbone (when comparing with the literature, the paper states that methods using
  pre-extracted features do not count the backbone).
- GFLOPs, peak memory and latency on the GPU (FP16 and FP32) and on the CPU with four threads, measured only with
  `benchmark_speed.py`.

## 7. Transparency: results seen before this plan was fixed

When this plan was fixed, the complete **seed-42** results (SumMe and TVSum) and part of the **seed-43** results on
SumMe had been seen. The primary model and the last-epoch rule were fixed before any result; the equivalence margins and
the primary metric were fixed after seed 42 and before the three seeds and the reference comparisons were complete.

## 8. Amendment: rule for selecting the primary model

The rule below was written and approved before the three seeds were complete, and its text was not changed afterwards.

**Rule:**
- Candidates: `mobilevitv2_050/proposed`, `mobilevit_xxs/proposed`, `mobilenetv3_small_100/proposed`.
- The primary model is the candidate with the **fewest parameters** that is **non-inferior** in τ (Otani) on TVSum to
  `mobilevitv2_050/proposed`: the lower bound of the 95% bootstrap confidence interval of the difference (10,000
  resamples, video level, after averaging the three seeds per video) must be **above −0.02**.
- If no smaller candidate meets this condition, `mobilevitv2_050/proposed` remains the primary model.
- SumMe is not used for this decision; its results are reported in full.
- The decision is taken after all three seeds are complete for all candidates, and is applied automatically in
  `summarize_results.py` (function `select_primary`, output `results/primary_selection.csv`).
- Candidates that are not selected are reported in the paper as alternative variants with their full results.

**Transparency:**
- The text of the rule was written before the results of MobileNetV3-Small with the correct surgery had been seen.
- When the rule was approved, the seed-42 results of MobileNetV3-Small (`proposed` and `surgery_only`, SumMe and TVSum,
  per fold) had appeared in the console; seeds 43 and 44 had not been trained yet.
- An earlier run of MobileNetV3-Small (`proposed` and `surgery_only`, seed 42) had been trained by mistake **without
  surgery** because of an incorrect version of `common.py`. Its results were deleted entirely and the models were
  retrained with the correct surgery (stem + first four blocks, 48 channels at 14 × 14); the invalid run was not used for
  any decision.

## 9. Pre-specified secondary experiment: does the temporal head pay off with denser frames? (60 frames)

Fixed before any 60-frame cache was built or any 60-frame model was trained.

**Motivation:** at 30 frames, `proposed` (τ = 0.1289) did not outperform `surgery_only` (τ = 0.1320) on TVSum, i.e. the
Conv1d-Transformer showed no benefit after surgery. Question: does a benefit appear with denser frames?

**Design (only the number of frames changes):**
- Backbone: `mobilevitv2_050` only. Configurations: `surgery_only` and `proposed` only. No other configuration is run in
  this experiment; any later addition is reported as exploratory.
- 60 frames instead of 30, uniformly spaced over the video (linspace).
- Unchanged: the same folds, `HPARAMS`, 30 epochs, last epoch, seeds (42, 43, 44) and evaluation at 150 points. KTS shots
  and annotator scores are copied verbatim from the 30-frame cache, so the evaluation is identical.
- 60 rather than 120 frames because of GPU memory during training (8 GB), decided before any result. If memory runs
  out, gradient checkpointing is used (it does not change the computation).
- Order: TVSum with the three seeds first, then SumMe.

**Hypotheses** (TVSum, τ following Otani, video level after averaging the three seeds):
- Δ30 = τ(proposed) − τ(surgery_only) at 30 frames (already known: −0.003).
- Δ60 = the same difference at 60 frames.
- **H1 (primary test):** Δ60 > 0, accepted if the lower bound of the 95% bootstrap interval of Δ60 (10,000 resamples)
  is above zero.
- **H2 (secondary):** Δ60 − Δ30 > 0, i.e. the benefit of the head grows with frame density; 95% interval of the
  video-level difference of differences.
- Descriptive only: τ of each configuration at 60 vs. 30 frames, F1 strict, SumMe results, and the cost (GFLOPs,
  latency, decoding) at 60 frames.

**Interpretation rule** (applied automatically in `analyze_frames60.py`):
- If H1 holds: the temporal head is justified at 60 frames, and the paper states that its benefit appears with denser
  frames, together with the cost.
- If H1 does not hold: the paper states explicitly that the temporal head showed no measurable benefit after surgery,
  at either 30 or 60 frames.
- The primary model selected in item 8 (at 30 frames) does not change with the outcome of this experiment.

**Transparency:** all 30-frame results (accuracy, speed and memory) were known when this item was fixed, including that
`surgery_only` was not worse than `proposed`. No 60-frame result existed.

## 10. Last pre-specified experiment: windowed inference over the whole video ("30 frames from every 100")

Fixed before any code for this experiment was written or run.

**Motivation:** a usage mode in which the model sees the whole video: the video is split into consecutive windows of
100 frames and 30 frames are taken from each window. This mode had not been evaluated; all accuracy figures refer to 30
frames spread over the whole video.

**Design (inference only, no retraining):**
- Model: `mobilevitv2_050/proposed` with the same trained checkpoints (last epoch, 3 seeds × 5 folds). Each video is
  evaluated only by the model of the fold in which it was a test video.
- For each video: consecutive, non-overlapping windows of 100 frames to the end of the video; 30 uniformly spaced frames
  from each window (the last, shorter window contributes up to 30 frames). Each window passes through the model on its
  own (30 frames, as in training).
- The scores of all sampled frames are gathered, linearly interpolated to the 150 evaluation points according to frame
  position, and evaluated exactly as before (same shots, annotators and protocol).
- Reason for not retraining in this mode: it would cost about 75 times the current training and need a cache of about
  17 GB for TVSum. The experiment therefore tests the **usage mode** on the model as trained.

**Comparison and decision** (TVSum, τ following Otani, video level after averaging the seeds):
- Δw = τ(windowed) − τ(30 uniformly spaced frames, existing results).
- **Non-inferiority:** lower bound of the 95% bootstrap interval of Δw (10,000 resamples) above −0.02 → the paper may
  describe windowed inference as a valid mode for covering the whole video.
- **Superiority:** lower bound above 0 → significant improvement.
- **Worse:** non-inferiority not met → the paper states that the model should be used as trained (30 frames spread over
  the video), and windowed inference remains future work requiring training in the same mode.
- Descriptive: F1 strict, SumMe, number of processed frames and time.

**Stopping rule:** this is the last experiment on the temporal head and on frame sampling. Any later experiment is
reported as exploratory only, and the result of this experiment is reported whatever it is.

**Transparency:** proposed after all previous results were known, including the 60-frame experiment (H1 not met).
Expectation before running: performance would most likely drop, because the model was trained on 30 widely spaced frames
(about 8 s apart) whereas a window covers only about 3 s, and scores from independent windows may not be comparable.

## 11. Addition: F1 with the standard protocol of the literature (protocol alignment only)

Fixed before any evaluation with this protocol was run.

- Sole purpose: to make F1 comparable with published methods in the literature table. No decision is based on it, and
  it does not change any conclusion of items 8–10.
- No training: the same checkpoints (last epoch, 3 seeds, 5 folds); each video is evaluated by the model of the fold in
  which it was a test video.
- The protocol is ported verbatim from `vsum_tools.py` and `knapsack.py` (KaiyangZhou/pytorch-vsumm-reinforce): model
  scores at the `picks` positions, shot value = mean, official change points, capacity floor(15% × n_frames), SumMe =
  maximum over users and TVSum = average. The ported knapsack was checked against the original on 300 random cases.
- All 16 configurations, random scores and the human reference are evaluated with the same protocol and all are
  reported.
- A remaining difference: the literature uses five random 80/20 splits, whereas this study uses a fixed five-fold
  split; this is stated in the table note.
- **Correction before any result was used:** the first run passed the raw model outputs (logits, mostly negative) to the
  official knapsack, which then selected no shots in many videos (F1 = 0 for about half of the SumMe videos). Published
  models output probabilities in [0, 1], so our outputs are passed through a sigmoid before evaluation. All results of
  the first run were discarded.
- **Note added after the results:** after the corrected run, a descriptive analysis was added in `analyze_official_f1.py`
  (per-video paired differences, bootstrap, Wilcoxon, Spearman correlation between F1 and τ across configurations). It
  was written after the table had been seen, so it is entirely exploratory and does not change any conclusion of items
  8–10.

## 12. Addition: peak RAM on the CPU (operational measurement, not an accuracy hypothesis)

Fixed before any run.

- Purpose: to support the deployment analysis (constant memory with streaming inference) on devices without a GPU.
  Memory only; it does not affect accuracy or any conclusion of items 8–11.
- Script: `benchmark_cpu_ram.py`. Models: v2-050 `proposed` and MobileNetV3-Small `pure`. Frame counts: 30, 60, 100,
  150, 300, 600, 1200 and 2400.
- Two modes: all frames in memory and one forward pass / streaming: 30 frames read at a time, only their features kept,
  then the temporal head applied to all features.
- FP32, four CPU threads, one separate Python process per point.
- Metric: peak working set during the forward pass minus the process memory after the model is loaded (private bytes
  are recorded as well).
- Pre-specified safety rule: the all-at-once mode is not run if the linear prediction from the last successful point
  (× 1.15) exceeds 80% of the available memory. Such points are recorded as "not run" with the predicted value and are
  not presented in the paper as measurements.
- All results are reported whatever they are.
