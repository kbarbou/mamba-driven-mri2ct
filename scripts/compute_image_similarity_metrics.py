"""
Computes voxel-level image quality metrics (MAE, PSNR, MS-SSIM) between
ground-truth CTs and synthetic CT predictions. Results are saved per case
to a JSON file in the experimental_results directory.
"""

import SimpleITK as sitk
import numpy as np
from typing import Optional
from skimage.metrics import peak_signal_noise_ratio
from skimage.util.arraycrop import crop
from scipy.signal import fftconvolve
from scipy.ndimage import uniform_filter
import time
from datetime import timedelta
import os
import json

# ── USER CONFIGURATION ────────────────────────────────────────────────────────
GROUND_TRUTH_DIR = os.path.expanduser("~/synthRAD2025_Task1_Train/Task1")
# GROUND_TRUTH_DIR = /set/path/to/synthRAD2025_Task1_Train/Task1
PREDICTIONS_DIR  = os.path.expanduser("~/predictions")
# PREDICTIONS_DIR = /set/path/to/predictions
JSON_OUTPUT_DIR  = os.path.expanduser("~/experimental_results")
# JSON_OUTPUT_DIR = /set/path/to/experimental_results
# ─────────────────────────────────────────────────────────────────────────────


class ImageMetrics:
    def __init__(self, debug=False):
        self.dynamic_range = [-1024., 3000.]
        self.debug = debug

    def score_patient(self, gt_img, synthetic_ct, mask):
        assert gt_img.shape == synthetic_ct.shape
        if mask is not None:
            assert mask.shape == synthetic_ct.shape

        gt_img       = np.clip(gt_img,       a_min=self.dynamic_range[0], a_max=self.dynamic_range[1])
        synthetic_ct = np.clip(synthetic_ct, a_min=self.dynamic_range[0], a_max=self.dynamic_range[1])

        ground_truth = gt_img       if mask is None else np.where(mask == 0, -1024, gt_img)
        prediction   = synthetic_ct if mask is None else np.where(mask == 0, -1024, synthetic_ct)

        _, ms_ssim_mask_value = self.ms_ssim(ground_truth, prediction, mask)

        return {
            "mae":     self.mae(ground_truth, prediction, mask),
            "psnr":    self.psnr(ground_truth, prediction, mask, use_population_range=True),
            "ms_ssim": ms_ssim_mask_value,
        }

    def mae(self, gt: np.ndarray, pred: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        if mask is None:
            mask = np.ones(gt.shape)
        else:
            mask = np.where(mask > 0, 1., 0.)
        return float(np.sum(np.abs(gt * mask - pred * mask)) / mask.sum())

    def psnr(self, gt: np.ndarray, pred: np.ndarray,
             mask: Optional[np.ndarray] = None,
             use_population_range: Optional[bool] = False) -> float:
        if mask is None:
            mask = np.ones(gt.shape)
        else:
            mask = np.where(mask > 0, 1., 0.)

        if use_population_range:
            gt   = np.clip(gt,   a_min=self.dynamic_range[0], a_max=self.dynamic_range[1])
            pred = np.clip(pred, a_min=self.dynamic_range[0], a_max=self.dynamic_range[1])
            dynamic_range = self.dynamic_range[1] - self.dynamic_range[0]
        else:
            dynamic_range = gt.max() - gt.min()
            pred = np.clip(pred, a_min=gt.min(), a_max=gt.max())

        return float(peak_signal_noise_ratio(gt[mask == 1], pred[mask == 1], data_range=dynamic_range))

    def structural_similarity_at_scale(self, im1, im2, *, luminance_weight=1,
                                       contrast_weight=1, structure_weight=1,
                                       win_size=None, data_range=None,
                                       gaussian_weights=False, full=False, **kwargs):
        K1 = kwargs.pop("K1", 0.01)
        K2 = kwargs.pop("K2", 0.03)
        sigma = kwargs.pop("sigma", 1.5)
        if K1 < 0: raise ValueError("K1 must be positive")
        if K2 < 0: raise ValueError("K2 must be positive")
        if sigma < 0: raise ValueError("sigma must be positive")
        use_sample_covariance = kwargs.pop("use_sample_covariance", True)

        if win_size is None:
            if gaussian_weights:
                truncate = 3.5
                r = int(truncate * sigma + 0.5)
                win_size = 2 * r + 1
            else:
                win_size = 7

        filter_func = uniform_filter
        filter_args = {"size": win_size}

        ndim = im1.ndim
        NP   = win_size ** ndim
        cov_norm = NP / (NP - 1) if use_sample_covariance else 1.0

        ux  = filter_func(im1, **filter_args)
        uy  = filter_func(im2, **filter_args)
        uxx = filter_func(im1 * im1, **filter_args)
        uyy = filter_func(im2 * im2, **filter_args)
        uxy = filter_func(im1 * im2, **filter_args)
        vx  = cov_norm * (uxx - ux * ux)
        vy  = cov_norm * (uyy - uy * uy)
        vxy = cov_norm * (uxy - ux * uy)

        vxsqrt = np.clip(vx, a_min=0, a_max=None) ** 0.5
        vysqrt = np.clip(vy, a_min=0, a_max=None) ** 0.5

        R  = data_range
        C1 = (K1 * R) ** 2
        C2 = (K2 * R) ** 2
        C3 = C2 / 2

        L      = np.clip((2 * ux * uy + C1) / (ux * ux + uy * uy + C1), a_min=0, a_max=None)
        C_comp = np.clip((2 * vxsqrt * vysqrt + C2) / (vx + vy + C2),   a_min=0, a_max=None)
        S      = np.clip((vxy + C3) / (vxsqrt * vysqrt + C3),           a_min=0, a_max=None)

        result = (L ** luminance_weight) * (C_comp ** contrast_weight) * (S ** structure_weight)
        pad    = (win_size - 1) // 2
        mssim  = crop(result, pad).mean(dtype=np.float64)

        return (mssim, result) if full else mssim

    def ms_ssim(self, gt: np.ndarray, pred: np.ndarray,
                mask: Optional[np.ndarray] = None,
                scale_weights: Optional[np.ndarray] = None) -> float:
        gt   = np.clip(gt,   min(self.dynamic_range), max(self.dynamic_range))
        pred = np.clip(pred, min(self.dynamic_range), max(self.dynamic_range))

        if mask is not None:
            mask = np.where(mask > 0, 1., 0.)
            gt   = np.where(mask == 0, min(self.dynamic_range), gt)
            pred = np.where(mask == 0, min(self.dynamic_range), pred)

        if min(self.dynamic_range) < 0:
            gt   = gt   - min(self.dynamic_range)
            pred = pred - min(self.dynamic_range)

        dynamic_range     = self.dynamic_range[1] - self.dynamic_range[0]
        scale_weights     = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]) if scale_weights is None else scale_weights
        luminance_weights = np.array([0, 0, 0, 0, 0, 0.1333]) if scale_weights is None else scale_weights
        levels            = len(scale_weights)
        downsample_filter = np.ones((2, 2, 2)) / 8

        target_size = 97
        pad_values  = [
            (np.clip((target_size - dim) // 2, a_min=0, a_max=None),
             np.clip(target_size - dim - (target_size - dim) // 2, a_min=0, a_max=None))
            for dim in gt.shape
        ]
        gt   = np.pad(gt,   pad_values, mode="edge")
        pred = np.pad(pred, pad_values, mode="edge")
        mask = np.pad(mask, pad_values, mode="edge")

        ms_ssim_vals, ms_ssim_maps = [], []
        for level in range(levels):
            _, ssim_map = self.structural_similarity_at_scale(
                gt, pred,
                luminance_weight=luminance_weights[level],
                contrast_weight=scale_weights[level],
                structure_weight=scale_weights[level],
                data_range=dynamic_range, full=True,
            )
            pad = 3
            ssim_masked = crop(ssim_map, pad)[crop(mask, pad).astype(bool)].mean(dtype=np.float64)
            ms_ssim_vals.append(self.structural_similarity_at_scale(
                gt, pred,
                luminance_weight=luminance_weights[level],
                contrast_weight=scale_weights[level],
                structure_weight=scale_weights[level],
                data_range=dynamic_range,
            ))
            ms_ssim_maps.append(ssim_masked)

            filtered = [fftconvolve(im, downsample_filter, mode="same") for im in [gt, pred]]
            gt, pred, mask = [x[::2, ::2, ::2] for x in [*filtered, mask]]

        ms_ssim_val      = np.prod([np.clip(x, 0, 1) for x in ms_ssim_vals])
        ms_ssim_mask_val = np.prod([np.clip(x, 0, 1) for x in ms_ssim_maps])
        return float(ms_ssim_val), float(ms_ssim_mask_val)


if __name__ == "__main__":
    os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)
    experiment_name  = os.path.basename(PREDICTIONS_DIR.rstrip("/"))
    json_output_path = os.path.join(JSON_OUTPUT_DIR, f"{experiment_name}_image_similarity_metrics.json")

    filenames = sorted(
        f.replace(".nii.gz", "")
        for f in os.listdir(PREDICTIONS_DIR) if f.endswith(".nii.gz")
    )

    metrics      = ImageMetrics()
    case_results = []
    total_start  = time.time()

    for filename in filenames:
        gt_path   = os.path.join(GROUND_TRUTH_DIR, f"{filename[1:3]}/{filename}/ct_{filename}.mha")
        pred_path = os.path.join(PREDICTIONS_DIR,  f"{filename}.nii.gz")
        mask_path = os.path.join(GROUND_TRUTH_DIR, f"{filename[1:3]}/{filename}/mask_{filename}.mha")

        if not os.path.exists(pred_path):
            print(f"Prediction not found: {pred_path}")
            continue

        gt   = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(gt_path)),   (2, 1, 0))
        pred = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(pred_path)), (2, 1, 0))
        mask = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(mask_path)), (2, 1, 0))

        print(f"GT shape: {gt.shape}  |  Pred shape: {pred.shape}")

        case_score = metrics.score_patient(gt, pred, mask)
        case_results.append({"case": filename, **case_score})
        print(f"Case: {filename}  {case_score}\n")

    metrics_arr = {k: [r[k] for r in case_results] for k in ("mae", "psnr", "ms_ssim")}
    print(f"MAE:     {np.mean(metrics_arr['mae']):.5f} ± {np.std(metrics_arr['mae']):.5f}")
    print(f"PSNR:    {np.mean(metrics_arr['psnr']):.5f} ± {np.std(metrics_arr['psnr']):.5f}")
    print(f"MS-SSIM: {np.mean(metrics_arr['ms_ssim']):.5f} ± {np.std(metrics_arr['ms_ssim']):.5f}")

    with open(json_output_path, "w") as f:
        json.dump(case_results, f, indent=2)
    print(f"\nMetrics saved to: {json_output_path}")

    print(f"Total time: {timedelta(seconds=int(time.time() - total_start))}")
