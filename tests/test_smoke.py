"""Smoke tests: ``poetry run pytest``."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from z_retina.asl import AsymmetricLossOptimised
from z_retina.evaluate import compute_all_metrics, find_optimal_thresholds_probs, probs_from_logits
from z_retina.model import CAFormerModel
from z_retina.patient_split import generate_patient_split
from z_retina.pseudo_patient import generate_pseudo_patient_split


def test_caformer_forward():
    m = CAFormerModel(num_classes=15, timm_pretrained=False, grad_checkpointing=False)
    m.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, 15)


def test_asl_and_metrics():
    m = CAFormerModel(num_classes=15, timm_pretrained=False, grad_checkpointing=False)
    m.eval()
    y = m(torch.randn(2, 3, 224, 224))
    crit = AsymmetricLossOptimised()
    loss = crit(y, torch.zeros(2, 15))
    assert loss.ndim == 0 and not torch.isnan(loss)
    logits = y.detach().numpy()
    labels = np.random.binomial(1, 0.1, (2, 15)).astype(np.float32)
    p = probs_from_logits(logits)
    thr = find_optimal_thresholds_probs(p, labels)
    met = compute_all_metrics(logits, labels, thr)
    assert "mAP" in met and "macro_F1" in met


def test_pseudo_patient_split_json():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        merged = {
            "annotations": {
                "im0.jpg": {"pathologies": ["Glaucoma"], "patient_id": "A"},
                "im1.jpg": {"pathologies": [], "patient_id": "A"},
                "im2.jpg": {"pathologies": ["Cataract"], "patient_id": "B"},
                "im3.jpg": {"pathologies": ["CHPRE"], "patient_id": "C"},
                "im4.jpg": {"pathologies": ["Glaucoma"], "patient_id": "D"},
                "im5.jpg": {"pathologies": [], "patient_id": "E"},
            }
        }
        mj = td / "merged.json"
        mj.write_text(json.dumps(merged), encoding="utf-8")
        out = td / "split.json"
        generate_pseudo_patient_split(
            merged_json=mj,
            out_json=out,
            infer_similarity_for_missing_ids=False,
            strict_patient_id=False,
            images_root=td,
            seed=0,
        )
        data = json.loads(out.read_text(encoding="utf-8"))
        n = len(data["train"]) + len(data["val"]) + len(data["test"])
        assert n == 6
        assert data.get("split_backend") == "group_shuffle"


def test_research_patient_split_json():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        merged = {
            "annotations": {
                "im0.jpg": {"pathologies": ["Glaucoma"], "patient_id": "A"},
                "im1.jpg": {"pathologies": [], "patient_id": "A"},
                "im2.jpg": {"pathologies": ["Cataract"], "patient_id": "B"},
                "im3.jpg": {"pathologies": ["CHPRE"], "patient_id": "C"},
                "im4.jpg": {"pathologies": ["Glaucoma"], "patient_id": "D"},
                "im5.jpg": {"pathologies": [], "patient_id": "E"},
            }
        }
        mj = td / "merged.json"
        mj.write_text(json.dumps(merged), encoding="utf-8")
        out = td / "split_research.json"
        generate_patient_split(
            merged_json=mj,
            out_json=out,
            infer_similarity_for_missing_ids=False,
            strict_patient_id=False,
            images_root=td,
            seed=0,
        )
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data.get("split_backend") == "research_ml_stratified"
        assert data.get("hash_method") == "dhash"
        n = len(data["train"]) + len(data["val"]) + len(data["test"])
        assert n == 6
