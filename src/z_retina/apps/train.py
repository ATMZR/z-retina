"""
Train CAFormer-B36 (Poetry entrypoint).

    poetry run z-retina-train --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from z_retina.asl import AsymmetricLossOptimised
from z_retina.dataset import CLASS_ORDER, MixCutCollator, build_datasets, make_weighted_sampler
from z_retina.model import CAFormerModel, load_caformer_checkpoint
from z_retina.splits import generate_split_from_config


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "bias" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    param_groups = [
        {"params": decay_params, "weight_decay": cfg["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=cfg["lr"], betas=tuple(cfg["betas"]))


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    warmup_steps = cfg["warmup_epochs"] * steps_per_epoch
    total_steps = cfg["total_epochs"] * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return cfg["min_lr"] / cfg["lr"] + 0.5 * (1.0 - cfg["min_lr"] / cfg["lr"]) * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_phase(epoch: int, cfg: dict) -> int:
    p1_end = cfg["phase1_epochs"]
    p2_end = p1_end + cfg["phase2_epochs"]
    if epoch < p1_end:
        return 1
    if epoch < p2_end:
        return 2
    return 3


def apply_phase(model: CAFormerModel, phase: int, cfg: dict) -> None:
    if phase == 1:
        model.freeze_backbone()
    elif phase == 2:
        model.unfreeze_last_n_blocks(cfg.get("phase2_blocks", 6))
    else:
        model.unfreeze_all()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    cfg: dict,
    epoch: int,
    ema=None,
) -> dict:
    model.train()
    total_loss = 0.0
    n_steps = 0
    grad_acc = cfg["grad_accumulation"]
    log_interval = cfg.get("log_interval", 20)
    precision = cfg.get("precision", "bf16")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    optimizer.zero_grad()

    for step, (images, labels) in enumerate(tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False)):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=dtype):
                logits = model(images)
        else:
            logits = model(images)
        logits_f = logits.float()
        labels_f = labels.float()
        loss = criterion(logits_f, labels_f) / grad_acc
        scaler.scale(loss).backward()
        if (step + 1) % grad_acc == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            if ema is not None:
                ema.update()
        total_loss += loss.item() * grad_acc
        n_steps += 1
        if (step + 1) % log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            tqdm.write(f"  step {step + 1}/{len(loader)} | loss {total_loss / n_steps:.6f} | lr {lr:.2e}")
    return {"loss": total_loss / max(n_steps, 1)}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg: dict,
    ema=None,
) -> dict:
    from z_retina.evaluate import compute_all_metrics

    if ema is not None:
        ema.store()
        ema.copy_to()
    model.eval()
    precision = cfg.get("precision", "bf16")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    all_logits, all_labels = [], []
    total_loss = 0.0
    for images, labels in tqdm(loader, desc="Val", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=dtype):
                logits = model(images)
        else:
            logits = model(images)
        loss = criterion(logits.float(), labels.float())
        total_loss += loss.item()
        all_logits.append(logits.float().cpu())
        all_labels.append(labels.float().cpu())
    if ema is not None:
        ema.restore()
    all_logits = torch.cat(all_logits, 0)
    all_labels = torch.cat(all_labels, 0)
    metrics = compute_all_metrics(all_logits.numpy(), all_labels.numpy())
    return {
        "val_loss": total_loss / len(loader),
        "val_mAP": metrics["mAP"],
        "val_f1_macro": metrics["macro_F1"],
        "val_f1_micro": metrics["micro_F1"],
        "val_mean_auc": metrics["mean_AUC"],
        "per_class_f1": metrics["per_class_F1"],
        "per_class_ap": metrics["per_class_AP"],
    }


def save_checkpoint(run_dir: Path, model, optimizer, scheduler, epoch: int, metrics: dict, cfg: dict, tag: str):
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": metrics,
            "config": cfg,
        },
        run_dir / f"{tag}.pth",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] {cfg.get('experiment', 'demo')}")
    print(f"[train] device={device} precision={cfg.get('precision', 'bf16')}")
    print(f"[train] patient_split_backend={cfg.get('patient_split_backend', 'research')}")

    generate_split_from_config(cfg)

    split_json = cfg.get("split_json")
    train_ds, val_ds, _ = build_datasets(
        dataset_root=Path(cfg["dataset_root"]),
        merged_json=Path(cfg["merged_json"]),
        n_channels=cfg["n_channels"],
        input_size=cfg["input_size"],
        cache_root=Path(cfg["cache_root"]),
        use_cache=cfg.get("use_cache", True),
        split_json=Path(split_json) if split_json else None,
        images_root=Path(cfg["images_root"]) if cfg.get("images_root") else None,
    )
    print(f"[train] train={len(train_ds)} val={len(val_ds)}")

    sampler = make_weighted_sampler(train_ds, cfg)
    collator = MixCutCollator(
        mixup_alpha=cfg.get("mixup_alpha", 0.4),
        mixup_prob=cfg.get("mixup_prob", 0.3),
        cutmix_alpha=cfg.get("cutmix_alpha", 1.0),
        cutmix_prob=cfg.get("cutmix_prob", 0.2),
    )
    nw = cfg.get("num_workers", 2)
    pf = cfg.get("prefetch_factor", 2) if nw > 0 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        sampler=sampler,
        num_workers=nw,
        prefetch_factor=pf,
        pin_memory=True,
        collate_fn=collator,
        persistent_workers=nw > 0,
    )
    nw_val = cfg.get("val_num_workers", 0)
    pf_val = cfg.get("prefetch_factor", 2) if nw_val > 0 else None
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=nw_val,
        prefetch_factor=pf_val,
        pin_memory=True,
        persistent_workers=False,
    )

    model = CAFormerModel(
        model_name=cfg.get("backbone", "caformer_b36"),
        num_classes=cfg["num_classes"],
        n_channels=cfg["n_channels"],
        drop_path_rate=cfg.get("drop_path_rate", 0.2),
        grad_checkpointing=cfg.get("grad_checkpointing", True),
        timm_pretrained=cfg.get("timm_pretrained", True),
    ).to(device)

    ema = None
    if cfg.get("use_ema", True):
        try:
            from torch_ema import ExponentialMovingAverage

            ema = ExponentialMovingAverage(model.parameters(), decay=cfg.get("ema_decay", 0.9998))
            print("[train] EMA on")
        except ImportError:
            print("[train] torch-ema not installed; EMA skipped")

    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = math.ceil(len(train_ds) / (cfg["batch_size"] * cfg["grad_accumulation"]))
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)

    pos_weight_tensor = None
    if cfg.get("asl_pos_weight", True):
        label_matrix = train_ds.get_label_matrix()
        pos_counts = label_matrix.sum(axis=0).clip(min=1)
        neg_counts = len(train_ds) - pos_counts
        pw = np.sqrt(neg_counts / pos_counts)
        boost_map = cfg.get("asl_pos_weight_boost_by_class") or {}
        for i, name in enumerate(CLASS_ORDER):
            if name in boost_map:
                pw[i] *= float(boost_map[name])
        pw_max = cfg.get("asl_pos_weight_max")
        if pw_max is not None:
            pw = np.minimum(pw, float(pw_max))
        pos_weight_tensor = torch.tensor(pw, dtype=torch.float32)

    criterion = AsymmetricLossOptimised(
        gamma_neg=cfg.get("asl_gamma_neg", 4.0),
        gamma_pos=cfg.get("asl_gamma_pos", 0.0),
        clip=cfg.get("asl_clip", 0.05),
        pos_weight=pos_weight_tensor,
    )
    precision = cfg.get("precision", "bf16")
    if device.type == "cuda":
        scaler = GradScaler("cuda", enabled=(precision == "fp16"))
    else:
        scaler = GradScaler("cpu", enabled=False)

    start_epoch = 0
    best_metric = cfg.get("best_metric", "val_mAP")
    if best_metric not in ("val_mAP", "val_f1_macro"):
        best_metric = "val_mAP"

    def _score(m: dict) -> float:
        v = m.get(best_metric, 0.0)
        try:
            x = float(v)
        except (TypeError, ValueError):
            return 0.0
        return x if not np.isnan(x) else 0.0

    best_score = 0.0
    resume_path = args.resume or cfg.get("resume_checkpoint")
    if resume_path and Path(resume_path).exists():
        print(f"[train] resume {resume_path}")
        ckpt = load_caformer_checkpoint(model, Path(resume_path), device)
        best_score = _score(ckpt.get("metrics") or {})
        if cfg.get("resume_optimizer", True) and "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
                scheduler.load_state_dict(ckpt["scheduler"])
                start_epoch = ckpt.get("epoch", -1) + 1
            except Exception as e:
                print(f"[train] optimizer load failed ({e}); fresh optimizer")
                start_epoch = 0
        else:
            start_epoch = 0
    elif resume_path:
        print(f"[train] WARNING missing resume file: {resume_path}")

    if start_epoch >= cfg["total_epochs"]:
        start_epoch = 0
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)

    log_path = run_dir / "training_log.csv"
    fields = ["epoch", "train_loss", "val_loss", "val_mAP", "val_f1_macro", "val_f1_micro", "val_mean_auc", "elapsed_s"]
    log_file = log_path.open("a", newline="")
    writer = csv.DictWriter(log_file, fieldnames=fields)
    if not log_path.exists() or start_epoch == 0:
        writer.writeheader()

    for epoch in range(start_epoch, cfg["total_epochs"]):
        apply_phase(model, get_phase(epoch, cfg), cfg)
        t0 = time.time()
        train_m = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, scaler, device, cfg, epoch, ema
        )
        val_m = validate(model, val_loader, criterion, device, cfg, ema)
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch + 1:03d}/{cfg['total_epochs']} | "
            f"loss={train_m['loss']:.4f} | val_mAP={val_m['val_mAP']:.4f} | "
            f"macroF1={val_m['val_f1_macro']:.4f} | {elapsed:.0f}s"
        )
        writer.writerow(
            {
                "epoch": epoch + 1,
                "train_loss": f"{train_m['loss']:.8f}",
                "val_loss": f"{val_m['val_loss']:.8f}",
                "val_mAP": f"{val_m['val_mAP']:.6f}",
                "val_f1_macro": f"{val_m['val_f1_macro']:.6f}",
                "val_f1_micro": f"{val_m['val_f1_micro']:.6f}",
                "val_mean_auc": f"{val_m['val_mean_auc']:.6f}",
                "elapsed_s": f"{elapsed:.1f}",
            }
        )
        log_file.flush()
        metrics = {**train_m, **val_m, "epoch": epoch}
        save_checkpoint(run_dir, model, optimizer, scheduler, epoch, metrics, cfg, "last")
        sc = _score(val_m)
        if sc > best_score:
            best_score = sc
            save_checkpoint(run_dir, model, optimizer, scheduler, epoch, metrics, cfg, "best")
            print(f"  -> best {best_metric}={best_score:.4f}")

    log_file.close()
    print(f"[train] done. best {best_metric}={best_score:.4f} | dir={run_dir}")


if __name__ == "__main__":
    main()
