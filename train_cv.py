"""
تدريب 5-fold بلا تسرّب: لا يُستخدم fold الاختبار في أي قرار أثناء التدريب.
يُحفظ checkpoint آخر epoch فقط (fold_k_last.pth) ويُقيَّم لاحقاً مرة واحدة.

مثال:
  python train_cv.py --dataset tvsum --backbone mobilevitv2_050 --config proposed --seed 42
اختبار سريع (fold واحد، epoch واحد):
  python train_cv.py --dataset summe --backbone mobilevitv2_050 --config proposed --seed 42 --folds 1 --epochs 1
"""
import os
import json
import time
import argparse

import numpy as np
import torch
import torch.optim as optim

import common as C


def train_one_fold(cache, train_idx, dataset, backbone, config, seed, epochs, out_path):
    C.seed_everything(seed)
    hp = C.HPARAMS[dataset]
    model = C.build_model(backbone, config, pretrained=True).to(C.DEVICE)
    C.enable_grad_checkpointing(model, backbone)

    criterion = C.HybridLoss(alpha=0.75, gamma=2.0, margin=0.1,
                             focal_weight=hp['focal_weight'], rank_weight=hp['rank_weight'])
    optimizer = optim.AdamW(model.parameters(), lr=C.LR, weight_decay=hp['weight_decay'])
    warmup = min(C.WARMUP_EPOCHS, max(epochs - 1, 1))
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup),
                    optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs - warmup, 1),
                                                         eta_min=hp['eta_min'])],
        milestones=[warmup])
    use_amp = C.DEVICE.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    rng = np.random.default_rng(seed)

    losses = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for i in rng.permutation(train_idx):
            frames = C.frames_tensor(cache[i]).unsqueeze(0).to(C.DEVICE)
            target = cache[i]['target'].unsqueeze(0).to(C.DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(frames)
                loss = criterion(logits.float(), target.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        scheduler.step()
        losses.append(total / len(train_idx))
        print(f"    epoch {epoch + 1:2d}/{epochs} | train loss {losses[-1]:.4f} | "
              f"lr {optimizer.param_groups[0]['lr']:.2e}", flush=True)

    torch.save(model.state_dict(), out_path)
    return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=C.DATASETS)
    ap.add_argument('--backbone', required=True, choices=C.ALL_BACKBONES)
    ap.add_argument('--config', required=True, choices=list(C.CONFIGS))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=C.EPOCHS)
    ap.add_argument('--folds', type=int, nargs='*', help='أرقام folds محددة (1..5)؛ الافتراضي كلها')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--tag', default='', help='لتشغيلات الاختبار (مثل smoke) حتى لا تختلط بالنتائج الحقيقية')
    args = ap.parse_args()

    cache = C.load_cache(args.dataset)
    folds = C.get_folds(len(cache))
    out_dir = C.run_dir(args.dataset, args.backbone, args.config, args.seed, args.tag)
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, 'train_log.json')
    log = json.load(open(log_path, encoding='utf-8')) if os.path.exists(log_path) else {}
    log['settings'] = dict(vars(args), hparams=C.HPARAMS[args.dataset], config_def=C.CONFIGS[args.config],
                           num_frames=C.NUM_FRAMES, lr=C.LR, warmup_epochs=C.WARMUP_EPOCHS,
                           checkpoint_selection='last epoch (no test-fold selection)')

    for k, (train_idx, _) in enumerate(folds, start=1):
        if args.folds and k not in args.folds:
            continue
        ckpt = os.path.join(out_dir, f"fold_{k}_last.pth")
        if os.path.exists(ckpt) and not args.overwrite:
            print(f"[skip] {ckpt} موجود")
            continue
        print(f"\n=== {args.dataset} | {args.backbone} | {args.config} | seed {args.seed} | fold {k}/{C.K_FOLDS} ===",
              flush=True)
        t0 = time.time()
        losses = train_one_fold(cache, train_idx, args.dataset, args.backbone, args.config,
                                args.seed, args.epochs, ckpt)
        log[f"fold_{k}"] = {'train_loss': losses, 'minutes': round((time.time() - t0) / 60, 2)}
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
