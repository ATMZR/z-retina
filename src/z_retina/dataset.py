"""
Fundus dataset (multi-label), on-disk preprocessing cache, and augmentations.

Training augmentations follow the production policy: geometry + mild
brightness/contrast/gamma + CLAHE + light noise/blur/compression + small
CoarseDropout — no hue shifts (colour is diagnostic).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from skimage.filters import frangi
from torch import Tensor
from torch.utils.data import Dataset, WeightedRandomSampler

# ---------------------------------------------------------------------------
# Class definitions (15 pathology labels; "Normal" = all-zero vector)
# ---------------------------------------------------------------------------

CLASS_ORDER = [
    "Atherosclerosis",
    "Cataract",
    "CHPRE",
    "Diabetic Retinopathy",
    "Glaucoma",
    "Hiv",
    "Hypertensive retinopathy",
    "Macular degeneration",
    "Malformation",
    "Peripheral Retinal Degeneration and Tear",
    "Pigmented Choroidal Neoplasm",
    "Retinitis Pigmentosa",
    "Retinoblastoma",
    "Systemic lupus",
    "Vascular Occulusions",
]
NUM_CLASSES = len(CLASS_ORDER)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Preprocessing (circular mask, Ben Graham, optional Frangi in 4-ch path)
# ---------------------------------------------------------------------------

def _find_circle(gray: np.ndarray) -> tuple[int, int, int]:
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = gray.shape
        return w // 2, h // 2, min(w, h) // 2
    largest = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    return int(cx), int(cy), int(radius)


def circular_mask(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cx, cy, r = _find_circle(gray)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    masked = cv2.bitwise_and(img, img, mask=mask)
    x1 = max(cx - r, 0)
    y1 = max(cy - r, 0)
    x2 = min(cx + r, img.shape[1])
    y2 = min(cy + r, img.shape[0])
    return masked[y1:y2, x1:x2]


def ben_graham(img: np.ndarray, sigma: int = 10) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    result = cv2.addWeighted(img, 4, blurred, -4, 128)
    return np.clip(result, 0, 255).astype(np.uint8)


def frangi_vessel(img: np.ndarray, sigmas: tuple = (1, 2, 3, 4)) -> np.ndarray:
    green = img[:, :, 1].astype(np.float32) / 255.0
    vessel = frangi(
        green,
        sigmas=sigmas,
        black_ridges=False,
        alpha=0.5,
        beta=0.5,
        gamma=15,
    )
    vmax = vessel.max()
    if vmax > 0:
        vessel = vessel / vmax
    return vessel.astype(np.float32)


def preprocess_3ch(img_path: Path) -> np.ndarray:
    retry_waits_s = (0.2, 0.5, 1.0, 2.0)
    img = None
    for attempt in range(len(retry_waits_s) + 1):
        img = cv2.imread(str(img_path))
        if img is not None:
            break
        if attempt < len(retry_waits_s):
            time.sleep(retry_waits_s[attempt])
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 1:
        img = np.concatenate([img, img, img], axis=-1)
    img = circular_mask(img)
    return ben_graham(img, sigma=10)


def preprocess_4ch(img_path: Path) -> np.ndarray:
    rgb = preprocess_3ch(img_path)
    vessel = frangi_vessel(rgb)
    vessel_255 = (vessel * 255).astype(np.float32)
    rgb_f = rgb.astype(np.float32)
    return np.concatenate([rgb_f, vessel_255[:, :, None]], axis=-1)


def get_cache_path(img_path: Path, split: str, n_channels: int, cache_root: Path) -> Path:
    tag = f"cache_{n_channels}ch"
    return cache_root / tag / split / (img_path.stem + ".npy")


def load_or_preprocess(
    img_path: Path,
    split: str,
    n_channels: int,
    cache_root: Path,
    use_cache: bool = True,
) -> np.ndarray:
    cache_path = get_cache_path(img_path, split, n_channels, cache_root)
    if use_cache and cache_path.exists():
        arr = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        return np.array(arr)
    arr = preprocess_3ch(img_path) if n_channels == 3 else preprocess_4ch(img_path)
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, arr)
    return arr


def get_small_cache_path(
    img_path: Path, split: str, n_channels: int, cache_root: Path, size: int = 224
) -> Path:
    return cache_root / f"cache_{n_channels}ch_s{size}" / split / (img_path.stem + ".npy")


def load_or_preprocess_small(
    img_path: Path,
    split: str,
    n_channels: int,
    cache_root: Path,
    size: int = 224,
    use_cache: bool = True,
) -> np.ndarray:
    small_path = get_small_cache_path(img_path, split, n_channels, cache_root, size)
    if use_cache and small_path.exists():
        arr = np.load(small_path, mmap_mode="r", allow_pickle=False)
        return np.array(arr)
    arr = load_or_preprocess(img_path, split, n_channels, cache_root, use_cache)
    if n_channels == 3:
        resized = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    else:
        rgb = arr[:, :, :3].astype(np.float32)
        vessel = arr[:, :, 3]
        rgb_r = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        vessel_r = cv2.resize(vessel, (size, size), interpolation=cv2.INTER_AREA)
        resized = np.concatenate([rgb_r, vessel_r[:, :, np.newaxis]], axis=-1).astype(np.float32)
    if use_cache:
        small_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(small_path, resized)
    return resized


# ---------------------------------------------------------------------------
# Augmentations (Albumentations)
# ---------------------------------------------------------------------------


def build_train_transforms(
    input_size: int = 224,
    n_channels: int = 3,
    include_resize: bool = True,
) -> A.Compose:
    additional_targets = {"image4": "image"} if n_channels == 4 else {}
    resize_ops = [A.Resize(input_size, input_size)] if include_resize else []
    return A.Compose(
        [
            *resize_ops,
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.10,
                rotate_limit=180,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=0.5,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.OneOf(
                [
                    A.Blur(blur_limit=3),
                    A.GaussNoise(std_range=(0.005, 0.03)),
                    A.ImageCompression(quality_range=(80, 95)),
                ],
                p=0.3,
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.CoarseDropout(
                num_holes_range=(1, 2),
                hole_height_range=(8, 12),
                hole_width_range=(8, 12),
                fill=0,
                p=0.2,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        additional_targets=additional_targets,
    )


def build_val_transforms(input_size: int = 224, include_resize: bool = True) -> A.Compose:
    resize_ops = [A.Resize(input_size, input_size)] if include_resize else []
    return A.Compose(
        [
            *resize_ops,
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
    )


# ---------------------------------------------------------------------------
# MixUp / CutMix collators (multi-label safe — soft label blend)
# ---------------------------------------------------------------------------


class MixUpCollator:
    def __init__(self, alpha: float = 0.4, prob: float = 0.3):
        self.alpha = alpha
        self.prob = prob

    def __call__(self, batch: list) -> tuple[Tensor, Tensor]:
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.stack(labels)
        if np.random.random() > self.prob:
            return images, labels
        lam = float(np.random.beta(self.alpha, self.alpha))
        idx = torch.randperm(images.size(0))
        return lam * images + (1.0 - lam) * images[idx], lam * labels + (1.0 - lam) * labels[idx]


class CutMixCollator:
    def __init__(self, alpha: float = 1.0, prob: float = 0.2):
        self.alpha = alpha
        self.prob = prob

    @staticmethod
    def _rand_bbox(H: int, W: int, lam: float):
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        x1 = int(np.clip(cx - cut_w // 2, 0, W))
        y1 = int(np.clip(cy - cut_h // 2, 0, H))
        x2 = int(np.clip(cx + cut_w // 2, 0, W))
        y2 = int(np.clip(cy + cut_h // 2, 0, H))
        return x1, y1, x2, y2

    def __call__(self, batch: list) -> tuple[Tensor, Tensor]:
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.stack(labels)
        if np.random.random() > self.prob:
            return images, labels
        lam = float(np.random.beta(self.alpha, self.alpha))
        idx = torch.randperm(images.size(0))
        B, C, H, W = images.shape
        x1, y1, x2, y2 = self._rand_bbox(H, W, lam)
        images = images.clone()
        images[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
        actual_lam = 1.0 - (x2 - x1) * (y2 - y1) / (H * W)
        return images, actual_lam * labels + (1.0 - actual_lam) * labels[idx]


class MixCutCollator:
    def __init__(
        self,
        mixup_alpha: float = 0.4,
        mixup_prob: float = 0.3,
        cutmix_alpha: float = 1.0,
        cutmix_prob: float = 0.2,
    ):
        self.mixup = MixUpCollator(alpha=mixup_alpha, prob=1.0)
        self.cutmix = CutMixCollator(alpha=cutmix_alpha, prob=1.0)
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob

    def __call__(self, batch: list) -> tuple[Tensor, Tensor]:
        r = np.random.random()
        if r < self.mixup_prob:
            return self.mixup(batch)
        if r < self.mixup_prob + self.cutmix_prob:
            return self.cutmix(batch)
        images, labels = zip(*batch)
        return torch.stack(images), torch.stack(labels)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def load_merged_annotations(json_path: Path) -> dict[str, list[str]]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    annotations = data["annotations"]
    return {fname: info.get("pathologies", []) for fname, info in annotations.items()}


def labels_to_vector(pathologies: list[str]) -> np.ndarray:
    vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    for p in pathologies:
        if p in CLASS_TO_IDX:
            vec[CLASS_TO_IDX[p]] = 1.0
    return vec


class EyeDataset(Dataset):
    def __init__(
        self,
        img_dir: Path,
        split: str,
        annotations: dict[str, list[str]],
        filenames: Optional[list[str]] = None,
        n_channels: int = 3,
        transform=None,
        input_size: int = 224,
        cache_root: Path = Path("cache"),
        use_cache: bool = True,
    ):
        self.img_dir = Path(img_dir)
        self.split = split
        self.n_channels = n_channels
        self.cache_root = Path(cache_root)
        self.use_cache = use_cache
        self.input_size = input_size
        self.transform = transform or build_val_transforms(input_size, include_resize=False)

        self.samples: list[tuple[Path, np.ndarray]] = []
        missing_file = 0
        skipped_undetected = 0
        if filenames is None:
            entries = list(annotations.items())
        else:
            entries = [(fn, annotations.get(fn, [])) for fn in filenames]

        for fname, pathologies in entries:
            img_path = self.img_dir / fname
            if not img_path.exists():
                missing_file += 1
                continue
            label = labels_to_vector(pathologies)
            if label.sum() == 0 and "Undetected" in pathologies:
                skipped_undetected += 1
                continue
            self.samples.append((img_path, label))

        if missing_file:
            print(f"[EyeDataset:{split}] {missing_file} entries without image file")
        if skipped_undetected:
            print(f"[EyeDataset:{split}] skipped {skipped_undetected} Undetected-only images")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        max_fallback = min(8, len(self.samples))
        for offset in range(max_fallback):
            cur = (idx + offset) % len(self.samples)
            img_path, label = self.samples[cur]
            try:
                arr = load_or_preprocess_small(
                    img_path,
                    self.split,
                    self.n_channels,
                    self.cache_root,
                    size=self.input_size,
                    use_cache=self.use_cache,
                )
                if self.n_channels == 3:
                    arr_u8 = arr.astype(np.uint8)
                    out = self.transform(image=arr_u8)
                    image = out["image"]
                else:
                    rgb = arr[:, :, :3].astype(np.uint8)
                    vessel = arr[:, :, 3]
                    v_u8 = (vessel * 255).clip(0, 255).astype(np.uint8)
                    v_3 = np.stack([v_u8, v_u8, v_u8], axis=-1)
                    out = self.transform(image=rgb, image4=v_3)
                    image = torch.cat([out["image"], out["image4"][0:1]], dim=0)
                return image, torch.from_numpy(label)
            except (FileNotFoundError, OSError, ValueError) as exc:
                print(f"[EyeDataset:{self.split}] load fail {img_path.name}: {exc}")
        raise RuntimeError(f"[EyeDataset:{self.split}] failed loading from index {idx}")

    def get_label_matrix(self) -> np.ndarray:
        return np.stack([s[1] for s in self.samples], axis=0)


def make_weighted_sampler(dataset: EyeDataset, cfg: Optional[dict[str, Any]] = None) -> WeightedRandomSampler:
    cfg = cfg or {}
    mode = cfg.get("weighted_sampler", "inverse_min")
    label_matrix = dataset.get_label_matrix()
    class_counts = label_matrix.sum(axis=0).astype(np.float64)
    class_counts_safe = np.maximum(class_counts, 1.0)
    if mode == "effective_number":
        beta = min(float(cfg.get("effective_number_beta", 0.9999)), 0.99999999)
        eff = (1.0 - np.power(beta, class_counts_safe)) / (1.0 - beta)
        class_weights = 1.0 / np.maximum(eff, 1e-8)
    else:
        class_weights = 1.0 / class_counts_safe

    sample_weights = np.zeros(len(dataset), dtype=np.float32)
    for i, (_, lab) in enumerate(dataset.samples):
        active = lab > 0
        if active.any():
            if mode == "effective_number":
                sample_weights[i] = float(class_weights[active].max())
            else:
                sample_weights[i] = float(1.0 / class_counts_safe[active].min())
        else:
            sample_weights[i] = float(1.0 / class_counts_safe.max())

    rb = float(cfg.get("rare_class_boost", 1.0))
    rmax = cfg.get("rare_class_max_count")
    if rb > 1.0 and rmax is not None:
        thr = float(rmax)
        for i, (_, lab) in enumerate(dataset.samples):
            active = lab > 0
            if active.any() and (class_counts[active] <= thr).any():
                sample_weights[i] *= rb

    rbc = cfg.get("rare_boost_by_class") or {}
    if rbc:
        for i, (_, lab) in enumerate(dataset.samples):
            extra = 1.0
            for ci in np.where(lab > 0)[0]:
                name = CLASS_ORDER[int(ci)]
                if name in rbc:
                    extra = max(extra, float(rbc[name]))
            if extra > 1.0:
                sample_weights[i] *= extra

    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(dataset),
        replacement=True,
    )


def build_datasets(
    dataset_root: Path,
    merged_json: Path,
    n_channels: int = 3,
    input_size: int = 224,
    cache_root: Path = Path("cache"),
    use_cache: bool = True,
    split_json: Optional[Path] = None,
    images_root: Optional[Path] = None,
) -> tuple[EyeDataset, EyeDataset, EyeDataset]:
    annotations = load_merged_annotations(merged_json)
    if split_json is not None:
        from z_retina.patient_split import load_split_json

        split_lists = load_split_json(split_json)
    else:
        split_lists = None

    if split_lists is None:
        train_dir = dataset_root / "train"
        val_dir = dataset_root / "val"
        test_dir = dataset_root / "test"
        train_files = val_files = test_files = None
    else:
        base = images_root if images_root is not None else dataset_root
        train_dir = val_dir = test_dir = base
        train_files = split_lists["train"]
        val_files = split_lists["val"]
        test_files = split_lists["test"]

    train_ds = EyeDataset(
        train_dir,
        "train",
        annotations,
        train_files,
        n_channels,
        build_train_transforms(input_size, n_channels, include_resize=False),
        input_size,
        cache_root,
        use_cache,
    )
    val_ds = EyeDataset(
        val_dir,
        "val",
        annotations,
        val_files,
        n_channels,
        build_val_transforms(input_size, include_resize=False),
        input_size,
        cache_root,
        use_cache,
    )
    test_ds = EyeDataset(
        test_dir,
        "test",
        annotations,
        test_files,
        n_channels,
        build_val_transforms(input_size, include_resize=False),
        input_size,
        cache_root,
        use_cache,
    )
    return train_ds, val_ds, test_ds
