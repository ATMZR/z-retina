"""
Patient-level, multi-label stratified split utilities.

The split can be built from:
  1) explicit patient id metadata (preferred), and/or
  2) pseudo-patient grouping from image similarity (fallback).

Pseudo grouping uses a two-stage pipeline:
  - dHash near-duplicate linking
  - optional ORB+RANSAC verification on vessel-enhanced maps
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from z_retina.dataset import CLASS_ORDER, CLASS_TO_IDX


DEFAULT_PATIENT_KEYS = (
    "patient_id",
    "patient",
    "patientId",
    "subject_id",
    "subjectId",
    "person_id",
    "case_id",
)


@dataclass
class ImageRecord:
    filename: str
    pathologies: list[str]
    patient_id: str | None


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _read_annotations(merged_json: Path) -> dict[str, Any]:
    with merged_json.open(encoding="utf-8") as f:
        data = json.load(f)
    return data["annotations"]


def _extract_patient_id(
    fname: str,
    info: dict[str, Any],
    patient_keys: tuple[str, ...],
    filename_regex: str | None,
    strict: bool,
) -> str | None:
    for key in patient_keys:
        val = info.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()

    if filename_regex:
        m = re.search(filename_regex, fname)
        if m:
            if m.groupdict():
                if "patient" in m.groupdict():
                    return str(m.group("patient"))
            return str(m.group(1))

    if strict:
        raise ValueError(
            f"Missing patient_id for '{fname}'. Add patient id metadata or set "
            "patient_id_regex in config."
        )

    # Missing patient id can be inferred later via image similarity.
    return None


def _build_records(
    annotations: dict[str, Any],
    patient_keys: tuple[str, ...],
    filename_regex: str | None,
    strict_patient_id: bool,
) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for fname, info in annotations.items():
        pathologies = info.get("pathologies", [])
        # Skip "Undetected-only" samples to stay consistent with EyeDataset.
        encoded = np.zeros(len(CLASS_ORDER), dtype=np.float32)
        for p in pathologies:
            idx = CLASS_TO_IDX.get(p)
            if idx is not None:
                encoded[idx] = 1.0
        if encoded.sum() == 0 and "Undetected" in pathologies:
            continue

        patient_id = _extract_patient_id(
            fname=fname,
            info=info,
            patient_keys=patient_keys,
            filename_regex=filename_regex,
            strict=strict_patient_id,
        )
        records.append(ImageRecord(fname, pathologies, patient_id))
    return records


def _dhash64(gray_u8: np.ndarray) -> int:
    small = cv2.resize(gray_u8, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = diff.flatten().astype(np.uint8)
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return int(out)


def _hamming64(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def _prepare_gray(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Mild contrast normalization improves hash/ORB stability across cameras.
    gray = cv2.equalizeHist(gray)
    return gray


def _prepare_vessel_enhanced(gray_u8: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_u8)


def _load_vessel_enhanced(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    gray = _prepare_gray(img)
    return _prepare_vessel_enhanced(gray)


def _orb_inliers(img1: np.ndarray, img2: np.ndarray, nfeatures: int = 500) -> int:
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    if len(matches) < 8:
        return 0
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if mask is None:
        return 0
    return int(mask.sum())


def _infer_pseudo_patient_ids_by_similarity(
    records: list[ImageRecord],
    images_root: Path,
    hash_hamming_threshold: int = 8,
    bucket_prefix_bits: int = 16,
    max_pairs_per_bucket: int = 5000,
    use_orb_verify: bool = True,
    orb_hamming_gate: int = 14,
    orb_inliers_threshold: int = 25,
) -> tuple[dict[str, str], dict[str, int]]:
    """
    Return mapping {filename: pseudo_patient_id} for records with missing IDs.

    Strategy:
      1) Build dHash per image and compare only inside hash-prefix buckets.
      2) Link if Hamming <= threshold.
      3) Optionally run ORB+RANSAC for wider hash gaps before linking.
    """
    missing_idx = [i for i, r in enumerate(records) if r.patient_id is None]
    if not missing_idx:
        return {}, {"missing_records": 0, "hash_links": 0, "orb_links": 0}

    uf = UnionFind(len(records))
    hashes: dict[int, int] = {}
    skipped_missing_file = 0

    # Fast pre-filter: hash only files that share byte size with at least one other.
    # This keeps runtime manageable on large datasets.
    size_groups: dict[int, list[int]] = {}
    idx_to_path: dict[int, Path] = {}
    for i in missing_idx:
        p = images_root / records[i].filename
        idx_to_path[i] = p
        if not p.exists():
            skipped_missing_file += 1
            continue
        try:
            size_groups.setdefault(p.stat().st_size, []).append(i)
        except OSError:
            skipped_missing_file += 1

    candidate_idx: list[int] = []
    for idxs in size_groups.values():
        if len(idxs) > 1:
            candidate_idx.extend(idxs)

    for i in candidate_idx:
        p = idx_to_path[i]
        img = cv2.imread(str(p))
        if img is None:
            skipped_missing_file += 1
            continue
        gray = _prepare_gray(img)
        hashes[i] = _dhash64(gray)

    if not hashes:
        return {}, {
            "missing_records": len(missing_idx),
            "missing_files": skipped_missing_file,
            "candidates_hashed": 0,
            "hash_links": 0,
            "orb_links": 0,
        }

    prefix_shift = max(0, 64 - bucket_prefix_bits)
    buckets: dict[int, list[int]] = {}
    for i, h in hashes.items():
        key = h >> prefix_shift
        buckets.setdefault(key, []).append(i)

    hash_links = 0
    orb_links = 0
    vessel_cache: dict[int, np.ndarray] = {}
    path_cache = {i: images_root / records[i].filename for i in hashes}

    for idxs in buckets.values():
        n = len(idxs)
        if n <= 1:
            continue
        pair_count = 0
        for a_pos in range(n):
            ia = idxs[a_pos]
            ha = hashes[ia]
            for b_pos in range(a_pos + 1, n):
                ib = idxs[b_pos]
                hb = hashes[ib]
                pair_count += 1
                if pair_count > max_pairs_per_bucket:
                    break
                ham = _hamming64(ha, hb)
                if ham <= hash_hamming_threshold:
                    uf.union(ia, ib)
                    hash_links += 1
                    continue
                if use_orb_verify and ham <= orb_hamming_gate:
                    if ia not in vessel_cache:
                        v = _load_vessel_enhanced(path_cache[ia])
                        if v is None:
                            continue
                        vessel_cache[ia] = v
                    if ib not in vessel_cache:
                        v = _load_vessel_enhanced(path_cache[ib])
                        if v is None:
                            continue
                        vessel_cache[ib] = v
                    inliers = _orb_inliers(vessel_cache[ia], vessel_cache[ib])
                    if inliers >= orb_inliers_threshold:
                        uf.union(ia, ib)
                        orb_links += 1
            if pair_count > max_pairs_per_bucket:
                break

    # Convert UF groups to pseudo patient IDs.
    root_to_pid: dict[int, str] = {}
    filename_to_pid: dict[str, str] = {}
    next_id = 1
    for i in missing_idx:
        if i not in hashes:
            continue
        root = uf.find(i)
        if root not in root_to_pid:
            root_to_pid[root] = f"pseudo_{next_id:06d}"
            next_id += 1
        filename_to_pid[records[i].filename] = root_to_pid[root]

    stats = {
        "missing_records": len(missing_idx),
        "missing_files": skipped_missing_file,
        "candidates_hashed": len(hashes),
        "hash_links": hash_links,
        "orb_links": orb_links,
        "n_pseudo_groups": len(root_to_pid),
    }
    return filename_to_pid, stats


def _group_targets(records: list[ImageRecord]) -> tuple[list[str], np.ndarray, dict[str, list[str]]]:
    groups: dict[str, list[ImageRecord]] = {}
    for r in records:
        groups.setdefault(r.patient_id, []).append(r)

    group_ids = sorted(groups.keys())
    Y = np.zeros((len(group_ids), len(CLASS_ORDER)), dtype=np.int32)
    group_to_files: dict[str, list[str]] = {}

    for gi, gid in enumerate(group_ids):
        files: list[str] = []
        for rec in groups[gid]:
            files.append(rec.filename)
            for p in rec.pathologies:
                ci = CLASS_TO_IDX.get(p)
                if ci is not None:
                    Y[gi, ci] = 1
        group_to_files[gid] = files

    return group_ids, Y, group_to_files


def _split_group_ids(
    group_ids: list[str],
    group_y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[set[str], set[str], set[str]]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    idx = np.arange(len(group_ids))
    splitter_test = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=seed
    )
    trainval_idx, test_idx = next(splitter_test.split(idx, group_y))

    rel_val_ratio = val_ratio / max(train_ratio + val_ratio, 1e-8)
    splitter_val = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=rel_val_ratio, random_state=seed
    )
    tr_rel, va_rel = next(
        splitter_val.split(trainval_idx, group_y[trainval_idx])
    )
    train_idx = trainval_idx[tr_rel]
    val_idx = trainval_idx[va_rel]

    train_g = {group_ids[i] for i in train_idx}
    val_g = {group_ids[i] for i in val_idx}
    test_g = {group_ids[i] for i in test_idx}
    return train_g, val_g, test_g


def generate_patient_split(
    merged_json: Path,
    out_json: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
    patient_id_keys: tuple[str, ...] = DEFAULT_PATIENT_KEYS,
    patient_id_regex: str | None = None,
    strict_patient_id: bool = True,
    infer_similarity_for_missing_ids: bool = False,
    images_root: Path | None = None,
    hash_hamming_threshold: int = 8,
    bucket_prefix_bits: int = 16,
    max_pairs_per_bucket: int = 5000,
    use_orb_verify: bool = True,
    orb_hamming_gate: int = 14,
    orb_inliers_threshold: int = 25,
) -> dict[str, Any]:
    annotations = _read_annotations(merged_json)
    records = _build_records(
        annotations=annotations,
        patient_keys=patient_id_keys,
        filename_regex=patient_id_regex,
        strict_patient_id=strict_patient_id,
    )
    if not records:
        raise ValueError("No valid records found in merged annotations.")

    # Optional pseudo-patient inference for records without explicit patient id.
    similarity_stats: dict[str, int] = {}
    if infer_similarity_for_missing_ids:
        if images_root is None:
            raise ValueError("images_root is required when infer_similarity_for_missing_ids=true")
        inferred, similarity_stats = _infer_pseudo_patient_ids_by_similarity(
            records=records,
            images_root=images_root,
            hash_hamming_threshold=hash_hamming_threshold,
            bucket_prefix_bits=bucket_prefix_bits,
            max_pairs_per_bucket=max_pairs_per_bucket,
            use_orb_verify=use_orb_verify,
            orb_hamming_gate=orb_hamming_gate,
            orb_inliers_threshold=orb_inliers_threshold,
        )
        for r in records:
            if r.patient_id is None and r.filename in inferred:
                r.patient_id = inferred[r.filename]

    # Remaining missing IDs: keep each image as its own pseudo patient.
    standalone_count = 0
    for r in records:
        if r.patient_id is None:
            r.patient_id = f"single_{Path(r.filename).stem}"
            standalone_count += 1

    group_ids, group_y, group_to_files = _group_targets(records)
    train_g, val_g, test_g = _split_group_ids(
        group_ids=group_ids,
        group_y=group_y,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    def collect(groups: set[str]) -> list[str]:
        out: list[str] = []
        for gid in sorted(groups):
            out.extend(group_to_files[gid])
        return sorted(out)

    split = {
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "n_images": len(records),
        "n_patients": len(group_ids),
        "n_singleton_fallback_ids": standalone_count,
        "patient_id_keys": list(patient_id_keys),
        "patient_id_regex": patient_id_regex,
        "infer_similarity_for_missing_ids": infer_similarity_for_missing_ids,
        "similarity_stats": similarity_stats,
        "split_backend": "research_ml_stratified",
        "hash_method": "dhash",
        "train": collect(train_g),
        "val": collect(val_g),
        "test": collect(test_g),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)
    return split


def load_split_json(split_json: Path) -> dict[str, list[str]]:
    with split_json.open(encoding="utf-8") as f:
        data = json.load(f)
    return {
        "train": data["train"],
        "val": data["val"],
        "test": data["test"],
    }
