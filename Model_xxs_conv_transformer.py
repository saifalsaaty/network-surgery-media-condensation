import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class TemporalPoolingModule(nn.Module):
    def __init__(self, stride=2):
        super(TemporalPoolingModule, self).__init__()
        self.pool = nn.AvgPool1d(kernel_size=stride, stride=stride)

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        x = self.pool(x)
        x = x.permute(0, 2, 1).contiguous()
        return x


# ---------------------------------------------------------
# 🔬 الوحدة الجديدة: Conv1d محلي ثم طبقة Transformer كاملة
# ---------------------------------------------------------
class ConvTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.4, kernel_size=3):
        super().__init__()
        # Conv1d محلي: نافذة صغيرة (kernel_size=3 = الإطار السابق + الحالي + التالي)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                              padding=kernel_size // 2, groups=1)
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv_dropout = nn.Dropout(dropout)

        # طبقة Transformer كاملة (Attention + FFN + Norm + Residual) — بلا تعديل
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )

    def forward(self, x):
        # x: (B, T, D)
        residual = x
        x_conv = x.transpose(1, 2)          # (B, D, T) — Conv1d يتوقّع القنوات في المحور الأوسط
        x_conv = self.conv(x_conv)
        x_conv = x_conv.transpose(1, 2)     # (B, T, D) — إعادة الترتيب الأصلي
        x_conv = self.conv_dropout(x_conv)
        x = self.conv_norm(residual + x_conv)   # وصلة تخطّي + تطبيع بعد Conv

        x = self.transformer_layer(x)           # طبقة Transformer كاملة كما هي
        return x


# ---------------------------------------------------------
# 🏆 المعمارية الرئيسية — نسخة Conv+Transformer (طبقتان)
# ---------------------------------------------------------
class EdgeVideoSummarizer(nn.Module):
    def __init__(self, feature_dim=64, nhead=2, num_layers=2, backbone_name='mobilevitv2_050',
                 pretrained_backbone=True, stride=1, truncate_backbone=True, conv_kernel_size=3):
        super(EdgeVideoSummarizer, self).__init__()

        self.backbone = timm.create_model(backbone_name, pretrained=pretrained_backbone,
                                          num_classes=0, global_pool='')

        if truncate_backbone:
            modules = []
            if hasattr(self.backbone, 'stem'):
                modules.append(self.backbone.stem)
            elif hasattr(self.backbone, 'conv_stem'):
                modules.append(self.backbone.conv_stem)

            if hasattr(self.backbone, 'stages'):
                stages_list = list(self.backbone.stages.children())[:-1]
                modules.extend(stages_list)
                print(f"[SURGERY] Truncated the last stage of {backbone_name}.")
            elif hasattr(self.backbone, 'blocks'):
                blocks_list = list(self.backbone.blocks.children())[:-1]
                modules.extend(blocks_list)
                print(f"[SURGERY] Truncated the last block of {backbone_name}.")

            self.backbone = nn.Sequential(*modules)

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_out = self.backbone(dummy_input)
            if dummy_out.dim() == 4:
                backbone_out_dim = dummy_out.shape[1]
            else:
                backbone_out_dim = dummy_out.shape[-1]

        print(f"[MODEL] Backbone Output Dimension: {backbone_out_dim} (raw shape: {tuple(dummy_out.shape)})")

        self.projection = nn.Linear(backbone_out_dim, feature_dim)

        self.stride = stride
        if self.stride > 1:
            self.temporal_pooling = TemporalPoolingModule(stride=stride)

        self.pos_encoder = PositionalEncoding(d_model=feature_dim)

        # 🔬 المتغيّر الجديد: num_layers وحدات Conv+Transformer بدل Transformer وحده
        self.blocks = nn.ModuleList([
            ConvTransformerBlock(
                d_model=feature_dim, nhead=nhead, dim_feedforward=feature_dim * 4,
                dropout=0.4, kernel_size=conv_kernel_size
            )
            for _ in range(num_layers)
        ])
        print(f"[MODEL] {num_layers} x ConvTransformerBlock(kernel_size={conv_kernel_size}) initialized.")

        self.pred_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(0)

        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        features = self.backbone(x)

        if features.dim() > 2:
            features = features.mean(dim=[-2, -1])

        features = self.projection(features)
        features = features.view(b, t, -1)

        if self.stride > 1:
            features = self.temporal_pooling(features)

        features = self.pos_encoder(features)

        for block in self.blocks:
            features = block(features)

        scores = self.pred_head(features)
        scores = scores.squeeze(-1)

        if self.stride > 1:
            scores = scores.unsqueeze(1)
            scores = F.interpolate(scores, size=t, mode='linear', align_corners=False)
            scores = scores.squeeze(1)

        return scores
