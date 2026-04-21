"""Auto-generate train/val/test filename lists from ``merged.json`` (patient-level, no leakage)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from z_retina.patient_split import DEFAULT_PATIENT_KEYS, generate_patient_split
from z_retina.pseudo_patient import generate_pseudo_patient_split


def generate_split_from_config(cfg: dict[str, Any]) -> None:
    """Write ``split_json`` if missing, using ``patient_split_backend`` from config.

    Backends:
      - ``research`` (default): same as ``new_eyes`` — dHash (+ optional ORB) and
        ``MultilabelStratifiedShuffleSplit`` over patient groups.
      - ``group_shuffle``: pHash + Frangi/ORB + ``GroupShuffleSplit`` (lighter stratification).
    """
    split_json = cfg.get("split_json")
    if not split_json or not cfg.get("auto_generate_patient_split", False):
        return
    split_path = Path(split_json)
    if split_path.exists():
        return

    backend = (cfg.get("patient_split_backend") or "research").lower().strip()
    common: dict[str, Any] = {
        "merged_json": Path(cfg["merged_json"]),
        "out_json": split_path,
        "train_ratio": cfg.get("train_ratio", 0.7),
        "val_ratio": cfg.get("val_ratio", 0.1),
        "test_ratio": cfg.get("test_ratio", 0.2),
        "seed": cfg.get("seed", 42),
        "patient_id_keys": tuple(cfg.get("patient_id_keys", list(DEFAULT_PATIENT_KEYS))),
        "patient_id_regex": cfg.get("patient_id_regex"),
        "strict_patient_id": cfg.get("strict_patient_id", True),
        "infer_similarity_for_missing_ids": cfg.get("infer_similarity_for_missing_ids", False),
        "images_root": Path(cfg["images_root"]) if cfg.get("images_root") else None,
        "hash_hamming_threshold": cfg.get("hash_hamming_threshold", 8),
        "bucket_prefix_bits": cfg.get("bucket_prefix_bits", 16),
        "max_pairs_per_bucket": cfg.get("max_pairs_per_bucket", 5000),
        "use_orb_verify": cfg.get("use_orb_verify", True),
        "orb_hamming_gate": cfg.get("orb_hamming_gate", 14),
        "orb_inliers_threshold": cfg.get("orb_inliers_threshold", 25),
    }

    print(f"[split] generating -> {split_path} (backend={backend})")
    if backend in ("research", "new_eyes", "ml_stratified"):
        generate_patient_split(**common)
    elif backend in ("group_shuffle", "demo", "pseudo"):
        generate_pseudo_patient_split(**common)
    else:
        raise ValueError(
            f"Unknown patient_split_backend={backend!r}. "
            "Use 'research' (default, matches new_eyes) or 'group_shuffle'."
        )
