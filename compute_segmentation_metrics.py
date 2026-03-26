"""
Runs TotalSegmentator on both ground-truth and synthetic CTs, then computes
segmentation metrics (DICE, HD95) per patient. Results are saved to a JSON
file in the experimental_results directory.
"""

import numpy as np
from typing import Optional
import os
import torch
import SimpleITK as sitk
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from totalsegmentator.python_api import totalsegmentator
import time
from tqdm import tqdm
from datetime import timedelta
import json

# ── USER CONFIGURATION ────────────────────────────────────────────────────────
GROUND_TRUTH_DIR = os.path.expanduser("~/synthRAD2025_Task1_Train/Task1")
# GROUND_TRUTH_DIR = /set/path/to/synthRAD2025_Task1_Train/Task1
PREDICTIONS_DIR  = os.path.expanduser("~/predictions")
# PREDICTIONS_DIR = /set/path/to/predictions
JSON_OUTPUT_DIR  = os.path.expanduser("~/experimental_results")
# JSON_OUTPUT_DIR = /set/path/to/experimental_results
# ─────────────────────────────────────────────────────────────────────────────


class SegmentationMetrics:
    def __init__(self, debug=False):
        self.debug         = debug
        self.dynamic_range = [-1024., 3000.]

        # TotalSegmentator classes — see https://github.com/wasserth/TotalSegmentator
        self.classes_to_use = {
            "AB": [
                2, 3, 5, 6,
                *range(10, 15),   # lungs
                *range(26, 51),   # vertebrae
                51, 79,
                *range(92, 116),  # ribs
                116,
            ],
            "HN": [
                15, 16, 17,
                *range(26, 51),   # vertebrae
                79, 90, 91,
            ],
            "TH": [
                2, 3, 5, 6,
                *range(10, 15),   # lungs
                *range(26, 51),   # vertebrae
                51, 79,
                *range(92, 116),  # ribs
                116,
            ],
        }

    def score_patient(self, gt_segmentation, sct_segmentation, patient_id, orientation=None):
        anatomy = patient_id[1:3].upper()
        assert sct_segmentation.shape == gt_segmentation.shape

        gt_seg   = gt_segmentation.cpu().detach()   if torch.is_tensor(gt_segmentation)  else torch.from_numpy(gt_segmentation).cpu().detach()
        pred_seg = sct_segmentation.cpu().detach()  if torch.is_tensor(sct_segmentation) else torch.from_numpy(sct_segmentation).cpu().detach()
        assert gt_seg.shape == pred_seg.shape

        spacing = orientation[0] if orientation is not None else None

        metrics = [
            {
                "name": "DICE",
                "f":    DiceMetric(include_background=True, reduction="mean", get_not_nans=False),
            },
            {
                "name":   "HD95",
                "f":      HausdorffDistanceMetric(include_background=True, reduction="mean", percentile=95, get_not_nans=False),
                "kwargs": {"spacing": spacing},
            },
        ]

        for c in self.classes_to_use[anatomy]:
            gt_tensor = (gt_seg == c).view(1, 1, *gt_seg.shape)
            if gt_tensor.sum() == 0:
                if self.debug:
                    print(f"No {c} in {patient_id}")
                continue
            est_tensor = (pred_seg == c).view(1, 1, *pred_seg.shape)
            for metric in metrics:
                metric["f"](est_tensor, gt_tensor, **metric.get("kwargs", {}))

        result = {}
        for metric in metrics:
            result[metric["name"]] = metric["f"].aggregate().item()
            metric["f"].reset()
        return result


if __name__ == "__main__":
    os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)
    experiment_name  = os.path.basename(PREDICTIONS_DIR.rstrip("/"))
    json_output_path = os.path.join(JSON_OUTPUT_DIR, f"{experiment_name}_seg_metrics.json")

    filenames = sorted(
        f.replace(".nii.gz", "")
        for f in os.listdir(PREDICTIONS_DIR) if f.endswith(".nii.gz")
    )

    print(f"Reading data from: {PREDICTIONS_DIR}")

    metrics      = SegmentationMetrics(debug=True)
    case_results = []
    total_start  = time.time()

    for idx, filename in enumerate(tqdm(filenames), 1):
        gt_path   = os.path.join(GROUND_TRUTH_DIR, f"{filename[1:3]}/{filename}/ct_{filename}.nii.gz")
        pred_path = os.path.join(PREDICTIONS_DIR,  f"{filename}.nii.gz")

        if not os.path.exists(pred_path):
            print(f"Prediction not found: {pred_path}")
            continue

        gt_seg_img   = totalsegmentator(gt_path,   task="total", ml=True, fast=True)
        pred_seg_img = totalsegmentator(pred_path, task="total", ml=True, fast=True)

        gt_seg   = np.transpose(gt_seg_img.get_fdata(),   (2, 1, 0))
        pred_seg = np.transpose(pred_seg_img.get_fdata(), (2, 1, 0))

        spacing     = tuple(float(s) for s in gt_seg_img.header.get_zooms()[::-1])
        orientation = (spacing, None, None)

        case_score = metrics.score_patient(gt_seg, pred_seg, patient_id=filename, orientation=orientation)
        case_results.append({"case": filename, **case_score})
        print(f"{idx} Case: {filename}  {case_score}\n")

    dice_vals = [r["DICE"] for r in case_results]
    hd95_vals = [r["HD95"] for r in case_results]
    print(f"DICE: {np.mean(dice_vals):.5f} ± {np.std(dice_vals):.5f}")
    print(f"HD95: {np.mean(hd95_vals):.5f} ± {np.std(hd95_vals):.5f}")

    with open(json_output_path, "w") as f:
        json.dump(case_results, f, indent=2)
    print(f"\nMetrics saved to: {json_output_path}")

    print(f"Total time: {timedelta(seconds=int(time.time() - total_start))}")
