"""
الإعدادات والأدوات المشتركة لكل التجارب (SumMe + TVSum).
كل أرقام الورقة يجب أن تخرج من هذه السكربتات فقط، حتى تكون كل الصفوف محسوبة
بنفس المعالجة المسبقة ونفس وصفة التدريب ونفس البروتوكول.
"""
import os
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# المسارات — عدّلها هنا فقط
# ============================================================
# Dataset root: set the environment variable VS_DATA_ROOT, or place the datasets in ./data (see data/README.md).
ROOT = os.environ.get('VS_DATA_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
PATHS = {
    'summe': dict(videos=os.path.join(ROOT, 'SumMe', 'videos'),
                  annotations=os.path.join(ROOT, 'SumMe', 'GT')),
    'tvsum': dict(videos=os.path.join(ROOT, 'TVSum', 'videos'),
                  annotations=os.path.join(ROOT, 'TVSum', 'annotations', 'ydata-tvsum50-anno.tsv')),
}
WORK_DIR = os.path.dirname(os.path.abspath(__file__))

# عدد الإطارات المُعايَنة لكل فيديو (مدخل النموذج). 30 = كل نتائج الورقة الأساسية.
# لا تغيّره هنا: تجربة الإطارات الأكثف (البند 9 في ANALYSIS_PLAN.md) يضبطها run_frames60.py
# عبر متغير البيئة VS_NUM_FRAMES، وتُكتب نتائجها في مجلدات منفصلة (cache_f60، runs_f60، results_f60)
# فلا تختلط أبداً بنتائج الـ 30 إطاراً.
NUM_FRAMES = int(os.environ.get('VS_NUM_FRAMES', '30'))
_FRAMES_SUFFIX = '' if NUM_FRAMES == 30 else f"_f{NUM_FRAMES}"
BASE_CACHE_DIR = os.path.join(WORK_DIR, "cache")  # cache الـ 30 إطاراً: مصدر اللقطات وتعليقات المقيّمين
CACHE_DIR = os.path.join(WORK_DIR, "cache" + _FRAMES_SUFFIX)
RUNS_DIR = os.path.join(WORK_DIR, "runs" + _FRAMES_SUFFIX)
RESULTS_DIR = os.path.join(WORK_DIR, "results" + _FRAMES_SUFFIX)

# ============================================================
# إعدادات ثابتة
# ============================================================
FINE_FRAMES = 150      # دقة التقييم (KTS و F1)
IMAGE_SIZE = 224
K_FOLDS = 5
SPLIT_SEED = 42        # ثابت: نفس الـ folds لكل الإعدادات وكل الـ seeds
EPOCHS = 30
WARMUP_EPOCHS = 3
LR = 1e-4
CAPACITY_RATIO = 0.15
SEEDS = [42, 43, 44]

# وصفة التدريب لكل مجموعة — منقولة من سكربتات النموذج المقترح الأصلية،
# وتُطبَّق الآن على كل الإعدادات الأربعة بلا استثناء. تُذكر في Implementation Details.
HPARAMS = {
    'summe': dict(focal_weight=2.0, rank_weight=1.0, weight_decay=5e-2, eta_min=1e-7, reduction='max'),
    'tvsum': dict(focal_weight=1.0, rank_weight=2.0, weight_decay=1e-2, eta_min=1e-6, reduction='avg'),
}

# تصميم 2×2 كامل: القطع × النمذجة الزمنية
CONFIGS = {
    'pure':          dict(truncate=False, num_layers=0),  # backbone كامل، بلا Transformer
    'surgery_only':  dict(truncate=True,  num_layers=0),  # backbone مقطوع، بلا Transformer
    'full_temporal': dict(truncate=False, num_layers=2),  # backbone كامل + Conv1d-Transformer
    'proposed':      dict(truncate=True,  num_layers=2),  # النموذج المقترح
}
BACKBONES = ['mobilevitv2_050', 'mobilevit_xxs']
DATASETS = ['summe', 'tvsum']

# Backbones مرجعية أثقل للمقارنة العادلة (الخيار أ): كاملة بلا قطع، بنفس الرأس الزمني
# (Conv1d-Transformer) ونفس وصفة التدريب ونفس البروتوكول — الإعداد full_temporal فقط.
# GoogLeNet هو الـ backbone الذي بُنيت عليه ميزات معظم الطرق المنشورة في Table 8 (VASNet، DSNet، DR-DSN...).
REFERENCE_BACKBONES = ['googlenet', 'resnet50', 'mobilenetv3_large_100', 'vit_base_patch16_224']
# عائلة CNN خفيفة بنفس تصميم 2×2 الكامل (الجراحة × الـ Transformer) لاختبار تعميم الجراحة خارج MobileViT
CNN_ABLATION_BACKBONES = ['mobilenetv3_small_100']
ALL_BACKBONES = BACKBONES + CNN_ABLATION_BACKBONES + REFERENCE_BACKBONES
GRAD_CHECKPOINT = {'vit_base_patch16_224'}  # لتوفير ذاكرة التدريب فقط؛ لا يغيّر الحسابات
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# أدوات عامة
# ============================================================
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_dir(dataset, backbone, config, seed, tag=''):
    return os.path.join(RUNS_DIR, dataset, backbone, config, f"seed{seed}" + (f"_{tag}" if tag else ""))


def backbone_normalization(backbone):
    """متوسط/انحراف التطبيع الذي دُرِّب به الـ backbone مسبقاً (من إعدادات timm نفسها)."""
    import timm
    from timm.data import resolve_data_config
    cfg = resolve_data_config({}, model=timm.create_model(backbone, pretrained=False))
    return tuple(float(v) for v in cfg['mean']), tuple(float(v) for v in cfg['std'])


class Summarizer(nn.Module):
    """
    يغلّف EdgeVideoSummarizer الأصلي دون تعديله:
      - يطبّع الإدخال بنفس تطبيع الـ backbone المُدرَّب مسبقاً (موحّد لكل المجموعات والإعدادات)
      - الإطارات تدخل بقيم [0, 1]
    """

    def __init__(self, net, mean, std):
        super().__init__()
        self.net = net
        self.register_buffer('mean', torch.tensor(mean).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor(std).view(1, 1, 3, 1, 1), persistent=False)

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(0)
        return self.net((x - self.mean) / self.std)


def reference_backbone(name, pretrained):
    """backbone مرجعي كامل يُخرج متجهاً واحداً لكل إطار (B, D)، مع تطبيعه الأصلي."""
    if name == 'googlenet':
        import torchvision
        if pretrained:
            m = torchvision.models.googlenet(weights=torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1)
        else:  # نفس البنية تماماً (transform_input=True كما في النسخة المُدرَّبة) ثم تُحمَّل الأوزان
            m = torchvision.models.googlenet(weights=None, aux_logits=False, init_weights=False,
                                             transform_input=True)
        m.fc = nn.Identity()
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        import timm
        from timm.data import resolve_data_config
        m = timm.create_model(name, pretrained=pretrained, num_classes=0)
        cfg = resolve_data_config({}, model=m)
        mean, std = tuple(float(v) for v in cfg['mean']), tuple(float(v) for v in cfg['std'])
    m.eval()
    with torch.no_grad():
        dim = m(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)).shape[-1]
    m.train()
    return m, int(dim), mean, std


def truncated_cnn_backbone(name, pretrained):
    """
    نفس مبدأ الجراحة في MobileViT مطبَّقاً على MobileNetV3 (timm): نحذف آخر مرحلة تخفض الدقة
    المكانية إلى 1/32 وكل ما بعدها (conv_head)، ونُبقي stem + المراحل حتى دقة 1/16.
    نقطة القطع تُحدَّد تلقائياً من أبعاد المخرجات، لا يدوياً.
    """
    import timm
    from timm.data import resolve_data_config
    m = timm.create_model(name, pretrained=pretrained, num_classes=0)
    cfg = resolve_data_config({}, model=m)
    blocks = list(m.blocks.children())
    m.eval()
    with torch.no_grad():
        x = m.bn1(m.conv_stem(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)))
        sizes = []
        for b in blocks:
            x = b(x)
            sizes.append(x.shape[-1])
    cut = max(i for i in range(1, len(sizes)) if sizes[i] < sizes[i - 1])  # آخر مرحلة بـ stride 2
    feat = nn.Sequential(m.conv_stem, m.bn1, *blocks[:cut])
    with torch.no_grad():
        dim = feat(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)).shape[1]
    feat.train()
    print(f"[SURGERY] {name}: kept stem + blocks[:{cut}] (output {dim} channels at {sizes[cut - 1]}x{sizes[cut - 1]})")
    return feat, int(dim), tuple(float(v) for v in cfg['mean']), tuple(float(v) for v in cfg['std'])


def build_model(backbone, config, pretrained=True):
    from Model_xxs_conv_transformer import EdgeVideoSummarizer
    cfg = CONFIGS[config]
    if backbone in REFERENCE_BACKBONES or backbone in CNN_ABLATION_BACKBONES:
        if cfg['truncate'] and backbone not in CNN_ABLATION_BACKBONES:
            raise ValueError("الـ backbones المرجعية تُستخدم كاملة فقط (pure أو full_temporal)")
        # نبني الجزء الزمني من الكلاس الأصلي نفسه، ثم نستبدل الـ backbone وطبقة الإسقاط
        net = EdgeVideoSummarizer(feature_dim=64, nhead=2, num_layers=cfg['num_layers'],
                                  backbone_name='mobilevitv2_050', pretrained_backbone=False,
                                  stride=1, truncate_backbone=True)
        if cfg['truncate']:
            feat, dim, mean, std = truncated_cnn_backbone(backbone, pretrained)
        else:
            feat, dim, mean, std = reference_backbone(backbone, pretrained)
        net.backbone = feat
        net.projection = nn.Linear(dim, 64)
    else:
        net = EdgeVideoSummarizer(feature_dim=64, nhead=2, num_layers=cfg['num_layers'],
                                  backbone_name=backbone, pretrained_backbone=pretrained,
                                  stride=1, truncate_backbone=cfg['truncate'])
        mean, std = backbone_normalization(backbone)
    if cfg['num_layers'] == 0:
        # بلا نمذجة زمنية: لا positional encoding (كما في Model_xxs_v2050_baseline الأصلي)،
        # حتى يبقى الـ baseline مكانياً خالصاً بلا معرفة بموقع الإطار.
        net.pos_encoder = nn.Identity()
    return Summarizer(net, mean, std)


def enable_grad_checkpointing(model, backbone):
    """للتدريب فقط (ViT-Base): نفس الحسابات بذاكرة أقل. لا أثر على MobileViT."""
    if backbone in GRAD_CHECKPOINT and hasattr(model.net.backbone, 'set_grad_checkpointing'):
        model.net.backbone.set_grad_checkpointing(True)


class HybridLoss(nn.Module):
    """نفس دالة الخسارة في سكربتات التدريب الأصلية (Focal + Margin Ranking)."""

    def __init__(self, alpha=0.75, gamma=2.0, margin=0.1, focal_weight=2.0, rank_weight=1.0):
        super().__init__()
        self.alpha, self.gamma, self.margin = alpha, gamma, margin
        self.focal_weight, self.rank_weight = focal_weight, rank_weight

    def forward(self, preds_logits, targets):
        p = preds_logits.view(-1)
        t = targets.view(-1)
        bce = F.binary_cross_entropy_with_logits(p, t, reduction='none')
        prob = torch.sigmoid(p)
        p_t = prob * t + (1 - prob) * (1 - t)
        focal = bce * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            focal = (self.alpha * t + (1 - self.alpha) * (1 - t)) * focal
        focal = focal.mean()

        pred_diff = prob.unsqueeze(0) - prob.unsqueeze(1)
        mask = ((t.unsqueeze(0) - t.unsqueeze(1)) > 0).float()
        rank = torch.relu(self.margin - pred_diff)
        n = mask.sum()
        rank = (rank * mask).sum() / n if n > 0 else torch.zeros((), device=p.device)
        return self.focal_weight * focal + self.rank_weight * rank


# ============================================================
# البيانات والـ cache
# ============================================================
def load_dataset(dataset):
    """يُستخدم فقط لقراءة التعليقات و get_fine_annotation (نفس كودك الأصلي)."""
    if dataset == 'summe':
        from summe_dataset import SumMeDataset
        ds = SumMeDataset(videos_dir=PATHS['summe']['videos'],
                          annotations_dir=PATHS['summe']['annotations'], num_frames=NUM_FRAMES)
        ds.video_files = sorted(ds.video_files)  # ترتيب ثابت على أي جهاز
    elif dataset == 'tvsum':
        from tvsum_dataset_proposed_updated import TVSumDataset
        ds = TVSumDataset(videos_dir=PATHS['tvsum']['videos'],
                          annotations_file=PATHS['tvsum']['annotations'], num_frames=NUM_FRAMES)
    else:
        raise ValueError(dataset)
    return ds


def video_name(ds, dataset, idx):
    if dataset == 'summe':
        return os.path.splitext(ds.video_files[idx])[0]
    return str(ds.grouped_annotations.iloc[idx]['video_id'])


def mean_ground_truth(ds, dataset, idx):
    """متوسط المُقيِّمين بطول الفيديو الكامل (نفس مصدر هدف التدريب في السكربتات الأصلية)."""
    if dataset == 'summe':
        from scipy.io import loadmat
        mat = loadmat(os.path.join(PATHS['summe']['annotations'], f"{video_name(ds, dataset, idx)}.mat"))
        return mat['gt_score'].squeeze().astype(np.float64)
    row = ds.grouped_annotations.iloc[idx]
    return np.mean([np.array([float(v) for v in s.split(',')]) for s in row['scores']], axis=0)


def training_target(gt_sampled, dataset):
    """نفس تعريف الهدف في كل مجموعة كما في السكربتات الأصلية."""
    if dataset == 'summe':
        return gt_sampled / gt_sampled.max() if gt_sampled.max() > 0 else gt_sampled
    return (gt_sampled - gt_sampled.min()) / (gt_sampled.max() - gt_sampled.min() + 1e-8)


def read_frames(video_path, indices, size=IMAGE_SIZE):
    """قراءة موحّدة لكل المجموعات: OpenCV → RGB → resize → uint8 (T, 3, H, W)."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames, last = [], None
    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, fr = cap.read()
        if ok:
            fr = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), (size, size))
            last = fr
        else:
            fr = last if last is not None else np.zeros((size, size, 3), np.uint8)
        frames.append(fr)
    cap.release()
    return np.stack(frames).transpose(0, 3, 1, 2)


def video_file(dataset, name):
    return os.path.join(PATHS[dataset]['videos'], f"{name}.mp4")


def build_cache(dataset):
    """
    يفك ترميز كل فيديو ويحسب لقطات KTS مرة واحدة فقط.
    بعدها يصبح التدريب والتقييم سريعين جداً ولا يتكرر فك الترميز في كل epoch.
    """
    import cv2
    from kts_corrected import kts_segment, frames_to_pooled_features

    out_dir = os.path.join(CACHE_DIR, dataset)
    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset(dataset)
    items = []
    for idx in range(len(ds)):
        name = video_name(ds, dataset, idx)
        path = os.path.join(out_dir, f"{name}.pt")
        base_path = os.path.join(BASE_CACHE_DIR, dataset, f"{name}.pt")
        if not os.path.exists(path) and CACHE_DIR != BASE_CACHE_DIR:
            # عدد إطارات مختلف: نفك ترميز الإطارات الجديدة فقط، وننسخ اللقطات وتعليقات المقيّمين
            # حرفياً من cache الـ 30 إطاراً، حتى يبقى التقييم متطابقاً تماماً بين التجربتين.
            if not os.path.exists(base_path):
                raise FileNotFoundError(f"cache الـ 30 إطاراً غير موجود: {base_path}")
            print(f"[cache {NUM_FRAMES}f] {dataset}: {idx + 1}/{len(ds)} {name}", flush=True)
            base = torch.load(base_path, weights_only=False)
            gt = mean_ground_truth(ds, dataset, idx)
            n = len(gt)
            if n != base['n_frames'] or base['name'] != name:
                raise ValueError(f"عدم تطابق مع cache الـ 30 إطاراً: {name}")
            indices = np.linspace(0, n - 1, NUM_FRAMES, dtype=int)
            frames = read_frames(video_file(dataset, name), indices)
            item = {k: base[k] for k in ('name', 'annotators', 'shots', 'fps', 'n_frames', 'duration_s')}
            item.update(frames_u8=torch.from_numpy(frames), frame_indices=indices.tolist(),
                        target=torch.tensor(training_target(gt[indices], dataset), dtype=torch.float32))
            torch.save(item, path)
        if not os.path.exists(path):
            print(f"[cache] {dataset}: {idx + 1}/{len(ds)} {name}", flush=True)
            gt = mean_ground_truth(ds, dataset, idx)
            n = len(gt)
            indices = np.linspace(0, n - 1, NUM_FRAMES, dtype=int)
            frames = read_frames(video_file(dataset, name), indices)
            content, annotators = ds.get_fine_annotation(idx, fine_frames=FINE_FRAMES)
            shots = kts_segment(frames_to_pooled_features(content))

            cap = cv2.VideoCapture(video_file(dataset, name))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            cap.release()

            torch.save({
                'name': name,
                'frames_u8': torch.from_numpy(frames),                 # (30, 3, 224, 224) uint8
                'frame_indices': indices.tolist(),
                'target': torch.tensor(training_target(gt[indices], dataset), dtype=torch.float32),
                'annotators': annotators.float(),                       # (U, 150) لكل مُقيِّم
                'shots': [(int(s), int(e)) for s, e in shots],
                'fps': float(fps), 'n_frames': int(n),
                'duration_s': (n / fps) if fps > 0 else float('nan'),
            }, path)
        items.append(path)

    split_file = os.path.join(out_dir, "splits.json")
    if not os.path.exists(split_file):
        names = [video_name(ds, dataset, i) for i in range(len(ds))]
        with open(split_file, 'w', encoding='utf-8') as f:
            json.dump({f"fold_{k + 1}": {'train': [names[i] for i in tr], 'test': [names[i] for i in te]}
                       for k, (tr, te) in enumerate(get_folds(len(ds)))}, f, indent=2, ensure_ascii=False)
    return items


def load_cache(dataset):
    return [torch.load(p, weights_only=False) for p in build_cache(dataset)]


def frames_tensor(item):
    """الإطارات بقيم [0, 1]؛ التطبيع يتم داخل النموذج."""
    return item['frames_u8'].float() / 255.0


def get_folds(n_videos):
    """
    مطابق حرفياً لـ sklearn KFold(n_splits=5, shuffle=True, random_state=42).split(...) — نفس الـ folds
    التي دُرِّبت بها كل النماذج (تحقّقنا من التطابق مع sklearn ومع توزيع الفيديوهات الفعلي على SumMe).
    كُتب بـ numpy فقط حتى لا يتوقف أي سكربت إن تعطّل استيراد sklearn على الجهاز.
    """
    idx = np.arange(n_videos)
    np.random.RandomState(SPLIT_SEED).shuffle(idx)
    sizes = np.full(K_FOLDS, n_videos // K_FOLDS, dtype=int)
    sizes[: n_videos % K_FOLDS] += 1
    folds, cur = [], 0
    for s in sizes:
        test = np.sort(idx[cur:cur + s])
        mask = np.ones(n_videos, dtype=bool)
        mask[test] = False
        folds.append((np.arange(n_videos)[mask], test))
        cur += s
    return folds
