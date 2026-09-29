# Network Surgery for Semantic Media Condensation

Code, fold splits, analysis plan and results for the paper

> **A Parameter- and Storage-Efficient Hybrid Framework for Semantic Media Condensation Using Network Surgery**
> Saif K. Jarallah, Sawsen Abdulhadi Mahmood, Mohammed Hazim Alkawaz

The model truncates the final stage of a MobileViT-v2-050 backbone (network surgery) and scores frames with a two-block
Conv1d-augmented Transformer encoder. All 16 configurations of the study (a 2 × 2 design of surgery × temporal head on
three lightweight backbones, plus four heavier reference backbones) share one pipeline, training recipe, fold split and
evaluation protocol, and are trained with three seeds under five-fold cross-validation.

Code comments are mostly in Arabic; function names, file names and outputs are in English.

## Repository structure

| Path | Content |
|---|---|
| `common.py` | Paths, hyperparameters (`HPARAMS`), fold split, frame cache, model builder, loss |
| `Model_xxs_conv_transformer.py` | Backbone truncation (network surgery) and the Conv1d-Transformer temporal encoder |
| `summe_dataset.py`, `tvsum_dataset_proposed_updated.py`, `dataset.py` | Dataset readers |
| `kts_corrected.py` | Kernel temporal segmentation (shots) |
| `eval_protocols.py`, `evaluation_utils_with_fixed.py` | Knapsack selection, F1 (strict / standard), Kendall's τ and Spearman's ρ |
| `train_cv.py`, `evaluate_cv.py` | Training (last-epoch checkpoints) and unified evaluation |
| `run_all.py` / `run_all.bat` | Main 2 × 2 experiments on MobileViT-v2-050 and MobileViT-XXS |
| `run_reference.py` | MobileNetV3-Small 2 × 2 and the four reference backbones |
| `run_frames60.py`, `analyze_frames60.py` | Pre-specified 60-frame experiment (plan item 9) |
| `run_windowed_eval.py` | Pre-specified windowed inference (plan item 10) |
| `eval_official_f1.py`, `analyze_official_f1.py` | F1 with the standard protocol of the benchmark files (plan item 11) |
| `benchmark_speed.py`, `run_memory_benchmark.py`, `benchmark_cpu_ram.py` | Latency, FLOPs, GPU memory and CPU RAM (plan item 12) |
| `summarize_results.py` | Summary tables, statistical tests and primary-model selection |
| `summe_human_tau.py` | Human agreement on SumMe |
| `make_qualitative_*.py` | Qualitative figures (temporal scores and Grad-CAM) |
| `condense_video.py` | Condenses any video with a trained model: condensed video, shot list, score plot and storyboard |
| `ANALYSIS_PLAN.md` | Pre-specified analysis plan |
| `splits/` | The five-fold splits used for every configuration and seed |
| `results*/` | All result files reported in the paper (per-video values, summaries, efficiency, figures) |

## Requirements

Tested on Windows with PyTorch 2.5.1 (CUDA 12.1) on an NVIDIA GeForce RTX 4060 Laptop GPU (8 GB).

```bash
pip install -r requirements.txt
```

## Data

The videos and annotations are not redistributed here; download them from the original sources:

- **SumMe** (Gygli et al., ECCV 2014): videos and `GT/*.mat` annotation files.
- **TVSum** (Song et al., CVPR 2015): videos and `ydata-tvsum50-anno.tsv`.

Place them as described in `data/README.md`, or set the environment variable `VS_DATA_ROOT` to another folder with the
same layout. For `eval_official_f1.py`, place `eccv16_dataset_summe_google_pool5.h5` and
`eccv16_dataset_tvsum_google_pool5.h5` (from the `KaiyangZhou/pytorch-vsumm-reinforce` repository) in `h5/`.

## Trained checkpoints

The trained checkpoints of the proposed model are attached to the [Releases](../../releases) page as
`checkpoints_proposed.zip`: MobileViT-v2-050 with network surgery and the Conv1d-Transformer, trained on TVSum and on
SumMe, seeds 42, 43 and 44, five folds each. Extract the archive in the root of this repository. It creates
`runs/<dataset>/mobilevitv2_050/proposed/seed<seed>/fold_<k>_last.pth`, the folder that `evaluate_cv.py` and
`condense_video.py` read. `checkpoints_mobilevitv2_050_ablation.zip` holds the other three configurations of the
2 × 2 design on MobileViT-v2-050 (`pure`, `surgery_only`, `full_temporal`) with the same datasets, seeds and folds, and
is extracted the same way. `SHA256SUMS.txt` on the same page verifies the downloads.

With the checkpoints and the datasets in place, the following command reproduces the per-video accuracy results of
the proposed model in `results/`, up to small numerical differences between GPUs; latency and memory depend on the
machine:

```bash
python evaluate_cv.py --dataset tvsum --backbone mobilevitv2_050 --config proposed --seed 42
```

## Reproducing the results

1. `run_smoke_test.py` — quick check (one fold, one epoch).
2. `run_all.py` — builds the frame cache, trains and evaluates the MobileViT configurations (resumable).
3. `run_reference.py` — MobileNetV3-Small and the reference backbones (resumable).
4. `summarize_results.py` — summary tables and tests in `results/`.
5. `benchmark_speed.py`, `benchmark_cpu_ram.py` — efficiency measurements (run with no other load on the machine).
6. `run_frames60.py`, `run_windowed_eval.py`, `eval_official_f1.py` — secondary analyses.
7. `make_qualitative_figures.py`, `make_qualitative_best.py`, `make_qualitative_human.py` — qualitative figures.

The fold split is generated deterministically (`common.get_folds`, equivalent to scikit-learn `KFold(n_splits=5,
shuffle=True, random_state=42)`) and is identical to the files in `splits/`.

## Condensing a video

`condense_video.py` applies a trained model to any video file, or to every video in a folder:

```bash
python condense_video.py --video my_video.mp4
python condense_video.py --video my_video.mp4 --seconds 60 --mode windowed
python condense_video.py --video folder_of_videos --out_dir condensed
```

The script follows Algorithm 1 of the paper:

1. It samples 30 frames over the whole video.
2. It scores them with the truncated backbone in chunks of B frames, then with the temporal encoder.
3. It segments the video into shots with KTS.
4. It selects shots with 0/1 knapsack under a 15% duration budget.

Shot segmentation and selection use the same settings and code as the evaluation. Outputs are written to `condensed/`:

| File | Content |
|---|---|
| `<name>_condensed.mp4` | The selected shots of the original video, in their original order, resolution and frame rate. The audio is kept if `ffmpeg` or `imageio-ffmpeg` is installed. |
| `<name>_condensed.json` | Every shot with its frames, times, score and selection, and the settings used |
| `<name>_scores.png` | Predicted importance over time, with the shot boundaries and the selected shots |
| `<name>_storyboard.jpg` | One key frame for each continuous part of the condensed video |

| Option | Meaning |
|---|---|
| `--ratio 0.15` | Duration budget as a fraction of the video (0.15, as in the paper) |
| `--seconds 60` | Duration budget in seconds instead of `--ratio` |
| `--mode uniform` | 30 frames spread over the whole video (default; the main setting of the paper) |
| `--mode windowed` | 30 frames from every window of 100 frames (Section 4.2), for denser coverage of long videos |
| `--dataset tvsum` | Use the models trained on TVSum (default) or on SumMe |
| `--fold 1` | Model used for videos outside the benchmarks; `all` averages the five fold models (not evaluated in the paper) |
| `--ckpt` | Checkpoint file(s), or a run folder, instead of the default `runs/...` folder |
| `--device cpu` | Run on the CPU in FP32 (the GPU uses FP16 mixed precision by default) |

Checkpoints are read from `runs/<dataset>/<backbone>/<config>/seed<seed>/fold_<k>_last.pth`: extract
`checkpoints_proposed.zip` from the Releases page in the repository root (see [Trained checkpoints](#trained-checkpoints)),
or train the models with `run_all.py`. When the input is a TVSum or SumMe video, the script automatically uses the
model of the fold in which that video was a test video, so the output never comes from a model trained on the same
video.

## License

MIT (see `LICENSE`), for the code and the trained checkpoints. The datasets are subject to their own licenses.
