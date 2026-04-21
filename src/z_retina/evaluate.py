"""
Evaluation with the **Table S1** protocol:

1. Run **8-fold test-time augmentation** (TTA) on validation -> logits.
2. **Per-class thresholds** = argmax F1 on validation (grid search on probabilities).
3. Run **8-fold TTA** on the test split with the same model.
4. Report **mAP**, **macro / micro F1**, **mean AUC**, **Hamming loss**, and per-class AUC/F1.

This matches the paper-style reporting used internally for supplementary Table S1.

Usage:

    poetry run z-retina-eval --config configs/default.yaml --checkpoint runs/demo_caformer_b36/best.pth --table_s1

Optional: ``--no_tta`` for faster, non-augmented inference (numbers will not match Table S1).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from z_retina.dataset import CLASS_ORDER, NUM_CLASSES, build_datasets
from z_retina.model import CAFormerModel

TTA_TRANSFORMS = [
    lambda x: x,
    lambda x: torch.flip(x, [3]),
    lambda x: torch.flip(x, [2]),
    lambda x: torch.rot90(x, 1, [2, 3]),
    lambda x: torch.rot90(x, 2, [2, 3]),
    lambda x: torch.rot90(x, 3, [2, 3]),
    lambda x: torch.flip(torch.rot90(x, 1, [2, 3]), [3]),
    lambda x: torch.flip(torch.rot90(x, 3, [2, 3]), [3]),
]


def probs_from_logits(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def compute_map(probs: np.ndarray, labels: np.ndarray) -> float:
    aps = []
    for c in range(labels.shape[1]):
        if labels[:, c].sum() == 0:
            continue
        aps.append(average_precision_score(labels[:, c], probs[:, c]))
    return float(np.mean(aps)) if aps else 0.0


def find_optimal_thresholds_probs(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    thresholds = np.arange(0.05, 0.96, 0.01)
    optimal = np.full(NUM_CLASSES, 0.5)
    for c in range(NUM_CLASSES):
        if labels[:, c].sum() == 0:
            continue
        best_f1, best_t = 0.0, 0.5
        for t in thresholds:
            f1 = f1_score(labels[:, c], probs[:, c] >= t, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        optimal[c] = best_t
    return optimal


def compute_all_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict:
    probs = probs_from_logits(logits)
    if thresholds is None:
        thresholds = np.full(NUM_CLASSES, 0.5)
    preds = (probs >= thresholds[None, :]).astype(int)
    mAP = compute_map(probs, labels)
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    micro_f1 = float(f1_score(labels, preds, average="micro", zero_division=0))
    h_loss = float(hamming_loss(labels, preds))
    per_class_auc: dict[str, float] = {}
    per_class_ap: dict[str, float] = {}
    per_class_f1: dict[str, float] = {}
    per_class_n_pos: dict[str, int] = {}
    for i, cls in enumerate(CLASS_ORDER):
        per_class_n_pos[cls] = int(labels[:, i].sum())
        if labels[:, i].sum() == 0:
            per_class_ap[cls] = float("nan")
            per_class_auc[cls] = float("nan")
            per_class_f1[cls] = float("nan")
            continue
        per_class_ap[cls] = float(average_precision_score(labels[:, i], probs[:, i]))
        if labels[:, i].sum() == len(labels):
            per_class_auc[cls] = float("nan")
        else:
            try:
                per_class_auc[cls] = float(roc_auc_score(labels[:, i], probs[:, i]))
            except Exception:
                per_class_auc[cls] = float("nan")
        per_class_f1[cls] = float(f1_score(labels[:, i], preds[:, i], zero_division=0))
    auc_vals = [v for v in per_class_auc.values() if not (isinstance(v, float) and np.isnan(v))]
    mean_auc = float(np.mean(auc_vals)) if auc_vals else float("nan")
    return {
        "mAP": mAP,
        "macro_F1": macro_f1,
        "micro_F1": micro_f1,
        "mean_AUC": mean_auc,
        "hamming_loss": h_loss,
        "per_class_AUC": per_class_auc,
        "per_class_AP": per_class_ap,
        "per_class_F1": per_class_f1,
        "per_class_n_pos": per_class_n_pos,
    }


@torch.no_grad()
def predict_with_tta(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    precision: str = "bf16",
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    all_logits, all_labels = [], []
    for images, labels in tqdm(loader, desc="TTA inference"):
        images = images.to(device)
        batch_logits = []
        for tfm in TTA_TRANSFORMS:
            x = tfm(images)
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=dtype):
                    logits = model(x)
            else:
                logits = model(x)
            batch_logits.append(logits.float().cpu())
        avg = torch.stack(batch_logits, 0).mean(0)
        all_logits.append(avg)
        all_labels.append(labels.float())
    return torch.cat(all_logits, 0).numpy(), torch.cat(all_labels, 0).numpy()


@torch.no_grad()
def predict_no_tta(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    precision: str = "bf16",
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    all_logits, all_labels = [], []
    for images, labels in tqdm(loader, desc="Inference"):
        images = images.to(device)
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=dtype):
                logits = model(images)
        else:
            logits = model(images)
        all_logits.append(logits.float().cpu())
        all_labels.append(labels.float())
    return torch.cat(all_logits, 0).numpy(), torch.cat(all_labels, 0).numpy()


def build_model_for_eval(cfg: dict, device: torch.device) -> CAFormerModel:
    model = CAFormerModel(
        model_name=cfg.get("backbone", "caformer_b36"),
        num_classes=cfg.get("num_classes", NUM_CLASSES),
        n_channels=cfg.get("n_channels", 3),
        drop_path_rate=cfg.get("drop_path_rate", 0.2),
        grad_checkpointing=False,
        timm_pretrained=False,
    ).to(device)
    return model


def evaluate_table_s1(config_path: str, checkpoint_path: str, use_tta: bool = True) -> dict:
    """Table S1: val TTA -> F1-max thresholds -> test TTA -> metrics."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_ds, test_ds = build_datasets(
        dataset_root=Path(cfg["dataset_root"]),
        merged_json=Path(cfg["merged_json"]),
        n_channels=cfg["n_channels"],
        input_size=cfg["input_size"],
        cache_root=Path(cfg["cache_root"]),
        use_cache=cfg.get("use_cache", True),
        split_json=Path(cfg["split_json"]) if cfg.get("split_json") else None,
        images_root=Path(cfg["images_root"]) if cfg.get("images_root") else None,
    )
    nw = cfg.get("num_workers", 2)
    pf = cfg.get("prefetch_factor", 2) if nw > 0 else None
    bs = cfg.get("batch_size", 4) * 2
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, prefetch_factor=pf, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, prefetch_factor=pf, pin_memory=True)

    model = build_model_for_eval(cfg, device)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(sd, strict=False)
    model.eval()

    precision = cfg.get("precision", "bf16")
    predict = predict_with_tta if use_tta else predict_no_tta
    val_logits, val_labels = predict(model, val_loader, device, precision)
    val_probs = probs_from_logits(val_logits)
    thr = find_optimal_thresholds_probs(val_probs, val_labels)
    test_logits, test_labels = predict(model, test_loader, device, precision)
    metrics = compute_all_metrics(test_logits, test_labels, thr)
    metrics["protocol"] = "Table_S1_val_F1max_thresholds_then_test"
    metrics["tta"] = use_tta
    metrics["per_class_thresholds"] = {CLASS_ORDER[i]: float(thr[i]) for i in range(NUM_CLASSES)}
    return metrics


def print_metrics(m: dict) -> None:
    print("\n" + "=" * 60)
    print("  Table S1 metrics (test set, val-tuned thresholds)")
    print("=" * 60)
    print(f"  TTA          : {m.get('tta', True)}")
    print(f"  mAP          : {m['mAP']:.4f}")
    print(f"  macro-F1     : {m['macro_F1']:.4f}")
    print(f"  micro-F1     : {m['micro_F1']:.4f}")
    print(f"  mean AUC     : {m['mean_AUC']:.4f}")
    print(f"  Hamming loss : {m['hamming_loss']:.4f}")
    print("\n  Per-class AUC / F1:")
    for cls in CLASS_ORDER:
        auc = m["per_class_AUC"].get(cls, float("nan"))
        f1v = m["per_class_F1"].get(cls, float("nan"))
        auc_s = f"{auc:.3f}" if not np.isnan(auc) else " n/a"
        f1_s = f"{f1v:.3f}" if not np.isnan(f1v) else " n/a"
        print(f"    {cls:<45} AUC={auc_s}  F1={f1_s}")


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        if np.isnan(x) or np.isinf(x):
            return None
        return x
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--table_s1",
        action="store_true",
        help="Val TTA -> F1-max t -> test TTA (paper Table S1).",
    )
    parser.add_argument("--no_tta", action="store_true")
    parser.add_argument("--out_json", default=None)
    args = parser.parse_args()

    use_tta = not args.no_tta
    if args.table_s1:
        m = evaluate_table_s1(args.config, args.checkpoint, use_tta=use_tta)
        print_metrics(m)
        if args.out_json:
            with open(args.out_json, "w", encoding="utf-8") as f:
                json.dump(_json_safe(m), f, indent=2)
            print(f"\n[eval] wrote {args.out_json}")
    else:
        parser.print_help()
        print("\nPass --table_s1 to run the Table S1 evaluation protocol.")


if __name__ == "__main__":
    main()
