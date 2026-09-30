"""
Condense a whole video with a trained model of this repository (demo and deployment script).

Pipeline (Algorithm 1 and Section IV-B of the paper):
  1. Frame sampling over the whole video
       uniform  (default, the main setting of the paper): K = 30 frames spread uniformly over the video
       windowed (Section IV-B): 30 frames from every consecutive window of 100 frames
  2. Scoring with the surgically truncated MobileViT-v2-050 and the two-block Conv1d-Transformer.
     The backbone processes B frames at a time (streaming) and only the pooled feature vectors are kept;
     the temporal encoder then runs once on the whole feature sequence.
  3. The scores are interpolated to 150 evaluation points and the video is split into shots with KTS.
  4. Shots are selected with 0/1 knapsack under a duration budget (15% by default).
  5. Outputs, in --out_dir:
       <name>_condensed.mp4   the condensed video: the selected shots of the original video, in their original
                              order, resolution and frame rate (with audio when ffmpeg is available)
       <name>_condensed.json  every shot with its frames, times, score and selection, and all settings
       <name>_scores.png      importance scores over time, with the selected shots shaded
       <name>_storyboard.jpg  one key frame for each continuous part of the condensed video

Steps 3-4 use the same settings and selection code as the evaluation of the paper (eval_protocols.py).

Examples
  python condense_video.py --video my_video.mp4
  python condense_video.py --video my_video.mp4 --ratio 0.10
  python condense_video.py --video my_video.mp4 --seconds 60
  python condense_video.py --video my_video.mp4 --mode windowed
  python condense_video.py --video folder_of_videos --out_dir condensed
  python condense_video.py --video data/TVSum/videos/i3wAGJaaktw.mp4

Checkpoints are read from runs/<dataset>/<backbone>/<config>/seed<seed>/fold_<k>_last.pth (written by
train_cv.py, or extracted from checkpoints_proposed.zip on the Releases page), or from --ckpt. If the video belongs to the benchmark named by --dataset, the model of the fold in
which that video was a test video is used, as in the paper, so the model has never seen it. For any other video,
--fold chooses the model (default 1); --fold all averages the five fold models, a deployment option that is not
evaluated in the paper.
"""
import os
import io
import re
import sys
import json
import time
import shutil
import argparse
import contextlib
import subprocess

# Must be set before NumPy is imported. With conda (MKL builds of NumPy/SciPy), NumPy's MKL would otherwise start its
# own copy of the Intel OpenMP runtime (libiomp5md.dll) next to the one of PyTorch, and the process stops with
# "OMP: Error #15" at the first NumPy matrix product after the model has run. The sequential MKL layer is a supported
# Intel setting: it only makes NumPy's own (tiny) matrix products single-threaded; PyTorch and the results are unchanged.
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import numpy as np   # noqa: E402
import cv2           # noqa: E402

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
if WORK_DIR not in sys.path:
    sys.path.insert(0, WORK_DIR)

import common as C                                         # noqa: E402
from eval_protocols import minmax                          # noqa: E402
from evaluation_utils_with_fixed import knapsack_summary   # noqa: E402

VIDEO_EXT = ('.mp4', '.avi', '.mkv', '.mov', '.webm', '.m4v', '.mpg', '.mpeg', '.wmv', '.flv')
WINDOW, PER_WINDOW = 100, 30   # windowed mode, identical to run_windowed_eval.py
SEEK_GAP = 200                 # read by seeking when the needed frames are on average further apart than this
KTS_GRID = 4                   # 4 x 4 average-pooled RGB descriptor, identical to kts_corrected.frames_to_pooled_features
THUMB_W = 256                  # width of the stored thumbnails (storyboard only)


# ============================================================
# Small helpers
# ============================================================
def fmt_time(seconds):
    s = int(round(max(0.0, float(seconds))))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def rel(path):
    """Path relative to the repository when it is inside it (keeps personal folders out of the JSON files)."""
    path = os.path.abspath(path)
    try:
        r = os.path.relpath(path, WORK_DIR)
    except ValueError:  # other drive on Windows
        return path
    return path if r.startswith('..') else r


def video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if n <= 0:  # the container does not store the frame count: count by decoding
        n = 0
        while cap.grab():
            n += 1
    cap.release()
    if not (np.isfinite(fps) and fps > 0):
        print("  [!] frame rate unknown; assuming 25 fps")
        fps = 25.0
    if n < 2:
        raise ValueError(f"video too short or unreadable: {path}")
    return dict(n_frames=n, fps=fps, width=w, height=h, duration_s=n / fps)


# ============================================================
# Frame sampling and reading
# ============================================================
def uniform_groups(n):
    """One sequence of K frames spread uniformly over the whole video, Eq. (8)."""
    return [np.linspace(0, n - 1, C.NUM_FRAMES, dtype=int)]


def windowed_groups(n):
    """Consecutive non-overlapping windows of 100 frames; up to 30 frames from each (run_windowed_eval.py)."""
    wins = []
    for s in range(0, n, WINDOW):
        e = min(s + WINDOW, n)
        k = min(PER_WINDOW, e - s)
        wins.append(np.unique(np.linspace(s, e - 1, k).round().astype(int)))
    return wins


def iter_frames(path, indices, sequential):
    """
    Yields (index, BGR frame) for sorted unique frame indices.
    seek       : like common.read_frames (the frames of the main experiments)
    sequential : like run_windowed_eval.py (reads through the video; used when many frames are needed)
    A frame that cannot be read is replaced by the last frame read, as in common.read_frames.
    """
    cap = cv2.VideoCapture(path)
    pos, last = 0, None
    for fi in indices:
        fi = int(fi)
        ok, fr = False, None
        if sequential:
            while pos < fi:
                cap.grab()
                pos += 1
            ok = cap.grab()
            pos += 1
            if ok:
                ok, fr = cap.retrieve()
        else:
            if fi != pos:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            pos = fi + 1
        if ok and fr is not None:
            last = fr
        else:
            fr = last
        yield fi, fr
    cap.release()


def model_frame(bgr):
    """Identical to common.read_frames: BGR -> RGB -> 224 x 224, uint8."""
    return cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (C.IMAGE_SIZE, C.IMAGE_SIZE))


def kts_descriptor(rgb224):
    """Identical to kts_corrected.frames_to_pooled_features: 4 x 4 average pooling of the [0, 1] RGB frame."""
    s = C.IMAGE_SIZE // KTS_GRID
    x = rgb224.astype(np.float32) / 255.0
    return x.reshape(KTS_GRID, s, KTS_GRID, s, 3).mean(axis=(1, 3)).transpose(2, 0, 1).reshape(-1)


def thumbnail(bgr):
    h, w = bgr.shape[:2]
    return cv2.resize(bgr, (THUMB_W, max(1, round(h * THUMB_W / w))), interpolation=cv2.INTER_AREA)


# ============================================================
# Model: checkpoints and streaming scoring (Algorithm 1)
# ============================================================
def benchmark_test_fold(dataset, name):
    """Fold in which a benchmark video was a test video, or None if the video is not in that benchmark."""
    for p in (os.path.join(WORK_DIR, 'splits', f"{dataset}_splits.json"),
              os.path.join(C.BASE_CACHE_DIR, dataset, 'splits.json')):
        if os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                splits = json.load(f)
            for fold, part in splits.items():
                if name in part['test']:
                    return int(fold.split('_')[-1])
            return None
    return None


def resolve_checkpoints(args, name):
    """Returns (checkpoint paths, note)."""
    test_fold = benchmark_test_fold(args.dataset, name)
    ck = args.ckpt or [C.run_dir(args.dataset, args.backbone, args.config, args.seed)]
    if len(ck) == 1 and os.path.isdir(ck[0]):
        run = ck[0]
        if test_fold is not None:
            folds = [test_fold]
            note = (f"'{name}' is a {args.dataset.upper()} video: using the model of fold {test_fold}, "
                    "in which it was a test video (as in the paper)")
        elif args.fold == 'all':
            folds = list(range(1, C.K_FOLDS + 1))
            note = "average of the five fold models (a deployment option not evaluated in the paper)"
        else:
            folds = [int(args.fold)]
            note = f"model of fold {folds[0]}"
        paths = [os.path.join(run, f"fold_{k}_last.pth") for k in folds]
    else:
        paths, note = list(ck), "checkpoint(s) given with --ckpt"
        if test_fold is not None:
            for p in paths:
                m = re.search(r'fold_(\d+)', os.path.basename(p))
                if m and int(m.group(1)) != test_fold:
                    print(f"  [!] '{name}' is a {args.dataset.upper()} video that the model in {os.path.basename(p)} "
                          f"was trained on; its test fold is {test_fold}")
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            "checkpoint not found: " + ", ".join(missing) +
            "\n    Download checkpoints_proposed.zip from the Releases page of the repository and extract it in the "
            "repository root,\n    train the models with run_all.py, or pass a checkpoint with --ckpt path/to/fold_1_last.pth")
    return paths, note


def load_model(path, backbone, config, device):
    import torch
    with contextlib.redirect_stdout(io.StringIO()):  # build_model prints the model summary
        model = C.build_model(backbone, config, pretrained=False)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    return model.to(device).eval()


class StreamingScorer:
    """
    Algorithm 1 of the paper. The frames of one sequence pass through the truncated backbone B at a time, and only
    their spatially pooled feature vectors are kept (Steps 3-7); the temporal encoder and the prediction head then
    run once on the whole feature sequence (Steps 8-10). The logits equal those of model(frames) in one pass; this
    is checked on every video in uniform mode.
    """

    def __init__(self, models, device, amp, chunk):
        import torch
        self.torch, self.models, self.device, self.amp, self.chunk = torch, models, device, amp, chunk
        for m in models:
            if getattr(m.net, 'stride', 1) != 1:
                raise ValueError("temporal stride > 1 is not used by any configuration of this study")
        self.model_s = 0.0
        self._reset()

    def _reset(self):
        self.buf, self.feats = [], [[] for _ in self.models]

    def _autocast(self):
        return self.torch.autocast(device_type=self.device.type, enabled=self.amp)

    def _sync(self):
        if self.device.type == 'cuda':
            self.torch.cuda.synchronize()

    def _to_tensor(self, frames):
        x = self.torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()
        return x.float().div(255.0).to(self.device)  # [0, 1], as common.frames_tensor

    def add(self, rgb224):
        self.buf.append(rgb224)
        if len(self.buf) == self.chunk:
            self._flush()

    def _flush(self):
        if not self.buf:
            return
        t0 = time.perf_counter()
        x = self._to_tensor(self.buf)
        with self.torch.no_grad(), self._autocast():
            for m, feats in zip(self.models, self.feats):
                h = m.net.backbone((x - m.mean[0]) / m.std[0])   # normalization of the backbone's pre-training
                if h.dim() > 2:
                    h = h.mean(dim=[-2, -1])                       # spatial mean, Eq. (2)
                feats.append(h)
        self._sync()
        self.model_s += time.perf_counter() - t0
        self.buf = []

    def finish(self):
        """Ends the current sequence; returns one logit vector per model."""
        self._flush()
        t0 = time.perf_counter()
        out = []
        with self.torch.no_grad(), self._autocast():
            for m, feats in zip(self.models, self.feats):
                net = m.net
                z = net.projection(self.torch.cat(feats)).unsqueeze(0)   # Eq. (3)
                z = net.pos_encoder(z)
                for block in net.blocks:                                 # Eqs. (4)-(6)
                    z = block(z)
                out.append(net.pred_head(z).squeeze(-1).float().view(-1).cpu().numpy())  # logits (Eq. 7 before σ)
        self._sync()
        self.model_s += time.perf_counter() - t0
        self._reset()
        return out

    def one_pass(self, frames):
        """model(frames) in one pass, as in evaluate_cv.py (used only for the check)."""
        x = self._to_tensor(frames).unsqueeze(0)
        with self.torch.no_grad(), self._autocast():
            return [m(x).float().view(-1).cpu().numpy() for m in self.models]


# ============================================================
# Scores -> shots -> selection (same settings as the evaluation)
# ============================================================
def interp_linear(x, size):
    """Same as F.interpolate(x.view(1, 1, -1), size, mode='linear', align_corners=False) in evaluate_cv.py."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 1:
        return np.full(size, x[0])
    src = np.maximum((np.arange(size) + 0.5) * (n / size) - 0.5, 0.0)
    i0 = np.minimum(np.floor(src).astype(int), n - 1)
    i1 = np.minimum(i0 + 1, n - 1)
    lam = src - i0
    return (1.0 - lam) * x[i0] + lam * x[i1]


def kts_segment_fast(features, ncp=None):
    """
    Vectorized form of kts_corrected.kts_segment: the same kernel, cost, number of change points (T // 3),
    dynamic programme and tie-breaking, but faster for many points. It returns the same shots, except where two
    segmentations have exactly the same cost (e.g. repeated identical frames); there the last-digit rounding of the
    kernel decides, as it does between BLAS libraries, and both results are optimal.
    """
    X = np.asarray(features, dtype=np.float64)
    T = X.shape[0]
    if T <= 1:
        return [(0, max(T, 1))]
    if ncp is None:
        ncp = max(3, T // 3)
    ncp = min(ncp, T - 1)
    K = np.einsum('id,jd->ij', X, X)   # = X @ X.T, computed without the BLAS/OpenMP library (see MKL note above)
    cumdiag = np.concatenate([[0.0], np.cumsum(np.diag(K))])
    P = np.zeros((T + 1, T + 1))
    P[1:, 1:] = np.cumsum(np.cumsum(K, axis=0), axis=1)

    a = np.arange(T + 1)[:, None]   # segment start j
    b = np.arange(T + 1)[None, :]   # segment end t
    length = b - a
    with np.errstate(divide='ignore', invalid='ignore'):
        seg_sum = P[b, b] - P[a, b] - P[b, a] + P[a, a]
        cost = (cumdiag[b] - cumdiag[a]) - seg_sum / length
    cost[length <= 0] = np.inf       # only j < t is a valid segment

    m = ncp + 1
    dp = np.full((m + 1, T + 1), np.inf)
    split = np.zeros((m + 1, T + 1), dtype=int)
    dp[0, 0] = 0.0
    cols = np.arange(T + 1)
    for k in range(1, m + 1):
        M = dp[k - 1, k - 1:][:, None] + cost[k - 1:, :]   # rows: j = k-1 .. T
        j_best = np.argmin(M, axis=0)                        # first minimum, like the strict '<' of the loops
        ts = cols[k:]
        dp[k, ts] = M[j_best[ts], ts]
        split[k, ts] = j_best[ts] + (k - 1)

    cps, t, k = [], T, m
    while k > 0:
        j = split[k, t]
        if k > 1:
            cps.append(int(j))
        t, k = j, k - 1
    bounds = [0] + sorted(set(cps)) + [T]
    return [(int(s), int(e)) for s, e in zip(bounds[:-1], bounds[1:]) if e > s]


def select_shots(scores, shots, ratio, protocol):
    """Same as eval_protocols.summary: min-max scores; shot value = sum (strict) or mean (standard); knapsack."""
    s = minmax(scores)
    agg = np.sum if protocol == 'strict' else np.mean
    values = np.array([agg(s[a:b]) for a, b in shots], dtype=np.float32)
    weights = np.array([b - a for a, b in shots], dtype=int)
    capacity = max(1, int(ratio * len(s)))
    selected = knapsack_summary(values, weights, capacity).astype(bool)
    return selected, values, capacity


def merge_segments(segs):
    out = []
    for a, b in sorted(segs):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(int(a), int(b)) for a, b in out]


# ============================================================
# Outputs
# ============================================================
def find_ffmpeg():
    exe = shutil.which('ffmpeg')
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def has_audio(ffmpeg, path):
    r = subprocess.run([ffmpeg, '-hide_banner', '-i', path], capture_output=True, text=True, errors='ignore')
    return 'Audio:' in (r.stderr or '')


def write_with_ffmpeg(ffmpeg, src, dst, segs, fps):
    audio = has_audio(ffmpeg, src)
    parts, pads = [], []
    for i, (a, b) in enumerate(segs):
        parts.append(f"[0:v]trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS[v{i}]")
        pad = f"[v{i}]"
        if audio:
            parts.append(f"[0:a]atrim=start={a / fps:.6f}:end={b / fps:.6f},asetpts=PTS-STARTPTS[a{i}]")
            pad += f"[a{i}]"
        pads.append(pad)
    parts.append(f"{''.join(pads)}concat=n={len(segs)}:v=1:a={int(audio)}[v]" + ("[a]" if audio else ""))
    graph = ';'.join(parts)
    if len(graph) > 30000:
        return False, audio, "too many segments for one command line"
    cmd = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-i', src, '-filter_complex', graph, '-map', '[v]']
    if audio:
        cmd += ['-map', '[a]', '-c:a', 'aac', '-b:a', '128k']
    cmd += ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', dst]
    r = subprocess.run(cmd, capture_output=True, text=True, errors='ignore')
    ok = r.returncode == 0 and os.path.isfile(dst) and os.path.getsize(dst) > 0
    return ok, audio, ('' if ok else (r.stderr or '').strip()[-400:])


def write_with_opencv(src, dst, segs, fps, size):
    w, h = size
    cap = cv2.VideoCapture(src)
    if not (w and h):
        ok, fr = cap.read()
        h, w = fr.shape[:2]
    wr = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if not wr.isOpened():
        cap.release()
        raise IOError(f"OpenCV cannot write {dst}")
    for a, b in segs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, a)
        for _ in range(b - a):
            ok, fr = cap.read()
            if not ok:
                break
            if fr.shape[1] != w or fr.shape[0] != h:
                fr = cv2.resize(fr, (w, h))
            wr.write(fr)
    cap.release()
    wr.release()


def save_plot(path, name, t_points, scores, shot_frames, selected, fps, duration, title_extra, t_scored=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter

    fig, ax = plt.subplots(figsize=(10, 3.8))
    for (a, b), sel in zip(shot_frames, selected):
        if sel:
            ax.axvspan(a / fps, b / fps, color='#f2b134', alpha=0.35, lw=0)
    for a, _ in shot_frames[1:]:
        ax.axvline(a / fps, color='#b0b0b0', lw=0.5, zorder=0)
    ax.plot(t_points, scores, color='#1f5fa8', lw=1.6)
    handles = [plt.Line2D([], [], color='#1f5fa8', lw=1.6, label='predicted importance'),
               Patch(color='#f2b134', alpha=0.5, label='selected shots'),
               plt.Line2D([], [], color='#b0b0b0', lw=0.8, label='shot boundaries (KTS)')]
    if t_scored is not None and len(t_scored) <= 120:
        ax.plot(t_scored, np.full(len(t_scored), 0.015), '|', color='#c0392b', ms=7, mew=1.2)
        handles.append(plt.Line2D([], [], color='#c0392b', marker='|', ls='', ms=7, mew=1.2,
                                  label='frames seen by the model'))
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_time(v)))
    ax.set_xlabel('time (m:ss)')
    ax.set_ylabel('importance (normalized)')
    ax.set_title(f"{name}: {title_extra}", fontsize=10)
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=len(handles),
              fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_storyboard(path, tiles, cols=6):
    """tiles: list of (BGR thumbnail, label)."""
    if not tiles:
        return False
    cols = min(cols, len(tiles))
    rows = int(np.ceil(len(tiles) / cols))
    th = max(t.shape[0] for t, _ in tiles)
    label_h, gap = 26, 6
    W = cols * THUMB_W + (cols + 1) * gap
    H = rows * (th + label_h) + (rows + 1) * gap
    sheet = np.full((H, W, 3), 255, np.uint8)
    for i, (img, label) in enumerate(tiles):
        r, c = divmod(i, cols)
        x = gap + c * (THUMB_W + gap)
        y = gap + r * (th + label_h + gap)
        sheet[y:y + img.shape[0], x:x + img.shape[1]] = img
        cv2.putText(sheet, label, (x + 4, y + th + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    return cv2.imwrite(path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


# ============================================================
# One video
# ============================================================
def condense(path, args, device, amp, model_cache, ffmpeg):
    name = os.path.splitext(os.path.basename(path))[0]
    info = video_info(path)
    n, fps, duration = info['n_frames'], info['fps'], info['duration_s']
    print(f"\n=== {os.path.basename(path)}: {fmt_time(duration)}, {n} frames, {fps:.2f} fps, "
          f"{info['width']}x{info['height']} ===")

    ratio = args.ratio if args.seconds is None else min(0.95, args.seconds / duration)
    ckpts, note = resolve_checkpoints(args, name)
    for p in ckpts:
        if p not in model_cache:
            model_cache[p] = load_model(p, args.backbone, args.config, device)
    models = [model_cache[p] for p in ckpts]
    print(f"[model] {args.backbone} / {args.config}, trained on {args.dataset.upper()}: {note}")

    # ---- 1-2. sampling, streaming scoring, KTS descriptors (one pass over the video) ----
    groups = uniform_groups(n) if args.mode == 'uniform' else windowed_groups(n)
    T = args.seg_points
    fine = np.linspace(0, n - 1, T, dtype=int)            # evaluation points, as in the evaluation
    needed = np.unique(np.concatenate(groups + [fine]))
    fine_set = set(fine.tolist())
    desc, thumbs, kept = {}, {}, []
    scorer = StreamingScorer(models, device, amp, args.chunk)
    group_scores, g, p = [], 0, 0
    blank = np.zeros((info['height'] or C.IMAGE_SIZE, info['width'] or C.IMAGE_SIZE, 3), np.uint8)
    n_scored = int(sum(len(x) for x in groups))
    how = (f"{n_scored} frames spread uniformly" if args.mode == 'uniform'
           else f"{n_scored} frames in {len(groups)} windows of {WINDOW}")
    print(f"[1/4] scoring {how} (streaming, B = {args.chunk}) on {device.type.upper()} "
          f"({'FP16 AMP' if amp else 'FP32'})", flush=True)
    t0 = time.perf_counter()
    sequential = n / len(needed) < SEEK_GAP   # same frames either way; only the reading speed differs
    for fi, bgr in iter_frames(path, needed, sequential):
        if bgr is None:
            bgr = blank
        rgb = model_frame(bgr)
        if fi in fine_set:
            desc[fi] = kts_descriptor(rgb)
            thumbs[fi] = thumbnail(bgr)
        while g < len(groups) and groups[g][p] == fi:
            scorer.add(rgb)
            if args.mode == 'uniform':
                kept.append(rgb)
            p += 1
            if p == len(groups[g]):
                group_scores.append(scorer.finish())
                g, p = g + 1, 0
    assert g == len(groups), "internal error: not every sampled frame was scored"
    t_analysis = time.perf_counter() - t0
    t_model = scorer.model_s

    check = None
    if args.mode == 'uniform':
        full = scorer.one_pass(kept)
        check = float(max(np.max(np.abs(a - b)) for a, b in zip(group_scores[0], full)))
        print(f"      check: streaming (Algorithm 1) vs. one-pass forward, max |d logit| = {check:.2e}")

    idx_all = np.concatenate(groups)
    per_model = []
    for mi in range(len(models)):
        logits = np.concatenate([gs[mi] for gs in group_scores])
        per_model.append(interp_linear(logits, T) if args.mode == 'uniform' else np.interp(fine, idx_all, logits))
    scores = minmax(np.mean([minmax(s) for s in per_model], axis=0))

    # ---- 3. shots ----
    t1 = time.perf_counter()
    shots = kts_segment_fast(np.stack([desc[int(f)] for f in fine]))
    ends = np.append(fine, n)                              # first frame of each point; the end of the video
    shot_frames = [(int(ends[a]), int(ends[b])) for a, b in shots]
    print(f"[2/4] shots: KTS on {T} points -> {len(shots)} shots")

    # ---- 4. selection ----
    selected, values, capacity = select_shots(scores, shots, ratio, args.protocol)
    if not selected.any():
        best = int(np.argmax([scores[a:b].mean() for a, b in shots]))
        selected[best] = True
        print("  [!] the budget is shorter than every shot; kept the single best shot")
    segs = merge_segments([sf for sf, s in zip(shot_frames, selected) if s])
    n_out = sum(b - a for a, b in segs)
    t_select = time.perf_counter() - t1
    print(f"[3/4] knapsack ({ratio * 100:.1f}% budget, {args.protocol}): {int(selected.sum())} of {len(shots)} shots, "
          f"{fmt_time(n_out / fps)} of {fmt_time(duration)} ({n_out / n * 100:.1f}%)")

    # ---- 5. outputs ----
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, name)
    out_video = base + '_condensed.mp4'
    t2 = time.perf_counter()
    writer, audio = 'opencv', False
    if ffmpeg:
        print("[4/4] writing the condensed video with ffmpeg", flush=True)
        ok, audio, err = write_with_ffmpeg(ffmpeg, path, out_video, segs, fps)
        if ok:
            writer = 'ffmpeg'
        else:
            audio = False
            print(f"  [!] ffmpeg failed ({err}); writing the video without audio with OpenCV")
    else:
        print("[4/4] writing the condensed video with OpenCV (no audio; install ffmpeg or "
              "'pip install imageio-ffmpeg' to keep the audio)", flush=True)
    if writer == 'opencv':
        write_with_opencv(path, out_video, segs, fps, (info['width'], info['height']))
    t_write = time.perf_counter() - t2

    outputs = {'video': os.path.basename(out_video)}
    t_points = fine / fps
    title = (f"{int(selected.sum())} of {len(shots)} shots selected, {fmt_time(n_out / fps)} of "
             f"{fmt_time(duration)} ({n_out / n * 100:.1f}%)")
    if not args.no_plot:
        try:
            save_plot(base + '_scores.png', name, t_points, scores, shot_frames, selected, fps, duration, title,
                      t_scored=idx_all / fps)
            outputs['plot'] = os.path.basename(base + '_scores.png')
        except Exception as e:  # matplotlib missing or failing must not stop the condensation
            print(f"  [!] plot not written: {e}")
    if not args.no_storyboard:
        tiles, run = [], []
        for i, sel in enumerate(list(selected) + [False]):   # runs of adjacent selected shots
            if sel:
                run.append(i)
            elif run:
                a, b = shots[run[0]][0], shots[run[-1]][1]
                k = a + int(np.argmax(scores[a:b]))          # highest-scoring point of the run
                fa, fb = shot_frames[run[0]][0], shot_frames[run[-1]][1]
                tiles.append((thumbs[int(fine[k])], f"{fmt_time(fa / fps)}-{fmt_time(fb / fps)}"))
                run = []
        if save_storyboard(base + '_storyboard.jpg', tiles):
            outputs['storyboard'] = os.path.basename(base + '_storyboard.jpg')

    record = {
        'input': os.path.basename(path),
        'video': {'n_frames': n, 'fps': round(fps, 4), 'width': info['width'], 'height': info['height'],
                  'duration_s': round(duration, 3)},
        'model': {'backbone': args.backbone, 'config': args.config, 'trained_on': args.dataset,
                  'checkpoints': [rel(p) for p in ckpts], 'note': note, 'device': device.type,
                  'precision': 'FP16 (AMP)' if amp else 'FP32'},
        'sampling': {'mode': args.mode, 'frames_scored': n_scored, 'sequences': len(groups), 'chunk_B': args.chunk,
                     'streaming_check_max_abs_diff': check},
        'segmentation': {'method': 'KTS', 'points': T, 'shots': len(shots)},
        'selection': {'method': '0/1 knapsack', 'protocol': args.protocol, 'budget_ratio': round(ratio, 4),
                      'capacity_points': capacity, 'selected_shots': int(selected.sum()),
                      'condensed_frames': int(n_out), 'condensed_duration_s': round(n_out / fps, 3),
                      'condensed_ratio': round(n_out / n, 4)},
        'outputs': dict(outputs, writer=writer, audio=bool(audio)),
        'processing_s': {'decode_and_preprocess': round(t_analysis - t_model, 3), 'model': round(t_model, 3),
                         'kts_and_selection': round(t_select, 3), 'write_video': round(t_write, 3)},
        'shots': [{'id': i, 'start_frame': fa, 'end_frame': fb, 'start_s': round(fa / fps, 3),
                   'end_s': round(fb / fps, 3), 'value': round(float(v), 4),
                   'mean_score': round(float(scores[a:b].mean()), 4), 'selected': bool(s)}
                  for i, ((a, b), (fa, fb), v, s) in enumerate(zip(shots, shot_frames, values, selected))],
        'scores': {'point_frames': fine.tolist(), 'normalized': [round(float(x), 4) for x in scores]},
    }
    with open(base + '_condensed.json', 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=1, ensure_ascii=False)

    print(f"[ok] {out_video}" + (" (with audio)" if audio else " (no audio)"))
    for k in ('plot', 'storyboard'):
        if k in outputs:
            print(f"     {os.path.join(args.out_dir, outputs[k])}")
    print(f"     {base + '_condensed.json'}")
    print(f"     time: decode {t_analysis - t_model:.1f} s, model {t_model:.2f} s, write {t_write:.1f} s")
    return dict(video=os.path.basename(path), duration=duration, condensed=n_out / fps, shots=len(shots),
                selected=int(selected.sum()))


# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="Condense a whole video with a trained model (network surgery + Conv1d-Transformer).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  python condense_video.py --video my_video.mp4\n"
               "  python condense_video.py --video my_video.mp4 --seconds 60 --mode windowed\n"
               "  python condense_video.py --video folder_of_videos --out_dir condensed")
    ap.add_argument('--video', required=True, help="video file, or a folder of videos")
    ap.add_argument('--out_dir', default=os.path.join(WORK_DIR, 'condensed'), help="output folder")
    ap.add_argument('--ratio', type=float, default=C.CAPACITY_RATIO,
                    help="duration budget as a fraction of the video (default 0.15, as in the paper)")
    ap.add_argument('--seconds', type=float, default=None,
                    help="duration budget in seconds instead of --ratio (e.g. 60 for about one minute)")
    ap.add_argument('--mode', choices=['uniform', 'windowed'], default='uniform',
                    help="uniform: 30 frames over the whole video (main setting of the paper); "
                         "windowed: 30 frames from every 100 frames (denser, Section IV-B)")
    ap.add_argument('--dataset', choices=C.DATASETS, default='tvsum', help="benchmark the model was trained on")
    ap.add_argument('--backbone', choices=C.ALL_BACKBONES, default='mobilevitv2_050')
    ap.add_argument('--config', choices=list(C.CONFIGS), default='proposed')
    ap.add_argument('--seed', type=int, default=42, choices=C.SEEDS)
    ap.add_argument('--fold', default='1', choices=[str(k) for k in range(1, C.K_FOLDS + 1)] + ['all'],
                    help="model used for videos outside the benchmark (default 1); 'all' averages the five")
    ap.add_argument('--ckpt', nargs='+', default=None,
                    help="checkpoint file(s), or a run folder with fold_<k>_last.pth files")
    ap.add_argument('--protocol', choices=['strict', 'standard'], default='strict',
                    help="shot value in the knapsack: sum (strict, default) or mean (standard) of the scores")
    ap.add_argument('--seg_points', type=int, default=C.FINE_FRAMES,
                    help="points for KTS and selection (default 150, as in the evaluation)")
    ap.add_argument('--chunk', type=int, default=30, help="frames per backbone pass, B (default 30)")
    ap.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    ap.add_argument('--fp32', action='store_true', help="disable mixed precision on the GPU")
    ap.add_argument('--no_ffmpeg', action='store_true', help="write the video with OpenCV (no audio)")
    ap.add_argument('--no_plot', action='store_true')
    ap.add_argument('--no_storyboard', action='store_true')
    args = ap.parse_args()

    if not 0 < args.ratio < 1:
        ap.error("--ratio must be between 0 and 1")
    if args.seconds is not None and args.seconds <= 0:
        ap.error("--seconds must be positive")
    if args.seg_points < 10 or args.chunk < 1:
        ap.error("--seg_points must be >= 10 and --chunk >= 1")

    if os.path.isdir(args.video):
        videos = sorted(os.path.join(args.video, f) for f in os.listdir(args.video)
                        if f.lower().endswith(VIDEO_EXT) and not os.path.splitext(f)[0].endswith('_condensed'))
        if not videos:
            ap.error(f"no videos in {args.video}")
    elif os.path.isfile(args.video):
        videos = [args.video]
    else:
        ap.error(f"not found: {args.video}")

    import torch
    if args.device == 'cuda' and not torch.cuda.is_available():
        ap.error("CUDA is not available")
    device = torch.device('cuda' if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available())
                          else 'cpu')
    amp = device.type == 'cuda' and not args.fp32
    ffmpeg = None if args.no_ffmpeg else find_ffmpeg()

    model_cache, done = {}, []
    for v in videos:
        try:
            done.append(condense(v, args, device, amp, model_cache, ffmpeg))
        except Exception as e:
            if len(videos) == 1:
                raise
            print(f"  [!] {os.path.basename(v)} skipped: {e}")

    if len(videos) > 1:
        print(f"\n=== {len(done)} of {len(videos)} videos condensed -> {args.out_dir} ===")
        for d in done:
            print(f"  {d['video']:<40} {fmt_time(d['duration']):>8} -> {fmt_time(d['condensed']):>7}"
                  f"   ({d['selected']}/{d['shots']} shots)")


if __name__ == '__main__':
    main()
