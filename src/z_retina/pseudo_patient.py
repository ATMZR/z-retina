"""
Pseudo-patient grouping and patient-level split without leaking near-duplicates.

Pipeline for images missing explicit ``patient_id`` metadata:
  1. **pHash** — 64-bit perceptual hash (DCT 8×8 on 32×32 luminance) for fast similarity.
  2. **Frangi** — vesselness on the green channel (ridge filter) to build a vessel map
     for geometry checks (same idea as the 4-channel preprocessing branch).
  3. **ORB + RANSAC** — match vessel maps; homography inlier count gates wide hash gaps.
  4. **Union–Find** — transitive closure of “same patient” pairs → pseudo-patient IDs.
  5. **GroupShuffleSplit** — 70 / 10 / 20 split by group so all images of one patient
     stay in one fold (no inter-fold leakage).

For images with metadata keys such as ``patient_id``, those strings are used as
groups directly. Remaining singletons get ``single_<stem>`` IDs.

This module is self-contained for the demo; for multilabel-stratified group splits
see the full research codebase.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from skimage.filters import frangi

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

    def union(self, a: int, b: int) -> None:
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
            if m.groupdict() and "patient" in m.groupdict():
                return str(m.group("patient"))
            return str(m.group(1))
    if strict:
        raise ValueError(
            f"Missing patient_id for '{fname}'. Add metadata, set patient_id_regex, "
            "or use strict_patient_id=false with infer_similarity_for_missing_ids."
        )
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
        encoded = np.zeros(len(CLASS_ORDER), dtype=np.float32)
        for p in pathologies:
            idx = CLASS_TO_IDX.get(p)
            if idx is not None:
                encoded[idx] = 1.0
        if encoded.sum() == 0 and "Undetected" in pathologies:
            continue
        pid = _extract_patient_id(fname, info, patient_keys, filename_regex, strict_patient_id)
        records.append(ImageRecord(fname, pathologies, pid))
    return records


def _prepare_gray(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)


def phash64(gray_u8: np.ndarray) -> int:
    """64-bit perceptual hash (DCT low-frequency sign vs median)."""
    x = cv2.resize(gray_u8, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(x)
    low = dct[:8, :8].copy()
    low[0, 0] = 0.0
    flat = low.flatten()
    med = float(np.median(flat))
    bits = flat > med
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return int(out)


def _hamming64(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def _frangi_map_u8(rgb_u8: np.ndarray) -> np.ndarray:
    green = rgb_u8[:, :, 1].astype(np.float32) / 255.0
    v = frangi(green, sigmas=(1, 2, 3, 4), black_ridges=False, alpha=0.5, beta=0.5, gamma=15.0)
    vmax = float(v.max()) or 1.0
    v = (v / vmax * 255.0).clip(0, 255).astype(np.uint8)
    return v


def _orb_inliers_ransac(img1: np.ndarray, img2: np.ndarray, nfeatures: int = 500) -> int:
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


def _infer_pseudo_patient_ids(
    records: list[ImageRecord],
    images_root: Path,
    hash_hamming_threshold: int = 8,
    bucket_prefix_bits: int = 16,
    max_pairs_per_bucket: int = 5000,
    use_orb_verify: bool = True,
    orb_hamming_gate: int = 14,
    orb_inliers_threshold: int = 25,
) -> tuple[dict[str, str], dict[str, int]]:
    missing_idx = [i for i, r in enumerate(records) if r.patient_id is None]
    if not missing_idx:
        return {}, {"missing_records": 0, "hash_links": 0, "orb_links": 0}

    uf = UnionFind(len(records))
    hashes: dict[int, int] = {}
    skipped_missing_file = 0
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
        hashes[i] = phash64(gray)

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
        buckets.setdefault(h >> prefix_shift, []).append(i)

    hash_links = orb_links = 0
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
                    img_a = cv2.imread(str(path_cache[ia]))
                    img_b = cv2.imread(str(path_cache[ib]))
                    if img_a is None or img_b is None:
                        continue
                    rgb_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
                    rgb_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)
                    if ia not in vessel_cache:
                        vessel_cache[ia] = _frangi_map_u8(rgb_a)
                    if ib not in vessel_cache:
                        vessel_cache[ib] = _frangi_map_u8(rgb_b)
                    if _orb_inliers_ransac(vessel_cache[ia], vessel_cache[ib]) >= orb_inliers_threshold:
                        uf.union(ia, ib)
                        orb_links += 1
            if pair_count > max_pairs_per_bucket:
                break

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


def _group_strings(records: list[ImageRecord]) -> list[str]:
    """One group label per record (patient id or pseudo or single_*)."""
    out: list[str] = []
    for r in records:
        assert r.patient_id is not None
        out.append(r.patient_id)
    return out


def _split_with_group_shuffle(
    filenames: list[str],
    groups: list[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    X = np.zeros((len(filenames), 1))
    g = np.asarray(groups)

    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    idx_all = np.arange(len(filenames))
    trainval_idx, test_idx = next(gss_test.split(X, groups=g))

    g_tv = g[trainval_idx]
    rel_val = val_ratio / max(train_ratio + val_ratio, 1e-8)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed + 17)
    X_tv = np.zeros((len(trainval_idx), 1))
    tr_rel, va_rel = next(gss_val.split(X_tv, groups=g_tv))

    train_idx = trainval_idx[tr_rel]
    val_idx = trainval_idx[va_rel]

    files = np.asarray(filenames)
    return (
        sorted(files[train_idx].tolist()),
        sorted(files[val_idx].tolist()),
        sorted(files[test_idx].tolist()),
    )


def generate_pseudo_patient_split(
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
    """Write ``train`` / ``val`` / ``test`` filename lists to ``out_json``."""
    annotations = _read_annotations(merged_json)
    records = _build_records(annotations, patient_id_keys, patient_id_regex, strict_patient_id)
    if not records:
        raise ValueError("No valid records in merged annotations.")

    similarity_stats: dict[str, int] = {}
    if infer_similarity_for_missing_ids:
        if images_root is None:
            raise ValueError("images_root is required when infer_similarity_for_missing_ids=True")
        inferred, similarity_stats = _infer_pseudo_patient_ids(
            records,
            images_root,
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

    standalone_count = 0
    for r in records:
        if r.patient_id is None:
            r.patient_id = f"single_{Path(r.filename).stem}"
            standalone_count += 1

    filenames = [r.filename for r in records]
    groups = _group_strings(records)
    train_f, val_f, test_f = _split_with_group_shuffle(
        filenames, groups, train_ratio, val_ratio, test_ratio, seed
    )

    unique_groups = sorted(set(groups))
    split: dict[str, Any] = {
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "n_images": len(records),
        "n_groups": len(unique_groups),
        "n_singleton_fallback_ids": standalone_count,
        "patient_id_keys": list(patient_id_keys),
        "patient_id_regex": patient_id_regex,
        "infer_similarity_for_missing_ids": infer_similarity_for_missing_ids,
        "similarity_stats": similarity_stats,
        "split_backend": "group_shuffle",
        "hash_method": "phash",
        "splitter": "GroupShuffleSplit",
        "train": train_f,
        "val": val_f,
        "test": test_f,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)
    return split


def main() -> None:
    import argparse

    from z_retina.patient_split import generate_patient_split

    p = argparse.ArgumentParser(description="Generate patient-level split JSON (CLI).")
    p.add_argument("--merged_json", type=Path, required=True)
    p.add_argument("--images_root", type=Path, required=True)
    p.add_argument("--out_json", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--backend",
        choices=("research", "group_shuffle"),
        default="research",
        help="research=new_eyes (dHash+MLSS); group_shuffle=pHash+GroupShuffleSplit.",
    )
    p.add_argument("--no_infer", action="store_true", help="Do not run similarity for missing patient_id.")
    args = p.parse_args()
    infer = not args.no_infer
    common = dict(
        merged_json=args.merged_json,
        out_json=args.out_json,
        seed=args.seed,
        images_root=args.images_root,
        infer_similarity_for_missing_ids=infer,
        strict_patient_id=False,
    )
    if args.backend == "research":
        generate_patient_split(**common)
    else:
        generate_pseudo_patient_split(**common)
    print(f"[split] wrote {args.out_json} backend={args.backend}")


if __name__ == "__main__":
    main()
