import argparse
import math
import os
import time
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

# Must happen before the first CUDA API call.
_gpu_id = os.environ.get("TRAINING_GPU")
if _gpu_id is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_id
    print(f"[INIT] CUDA_VISIBLE_DEVICES set to {_gpu_id} (from TRAINING_GPU)")

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from monai.losses import SSIMLoss
from monai.utils import set_determinism

from light_training.dataloading.dataset import get_train_val_test_loader_from_train
from light_training.prediction import Predictor
from light_training.trainer import Trainer

set_determinism(123)

env        = "pytorch"
max_epoch  = 1
batch_size = 1
val_every  = 1
num_gpus   = 1
device     = "cuda:0"
patch_size =  [64, 224, 224]
# patch_size = [64, 192, 192]


def build_model(model_type: str, use_cuda: bool, depths=None, feat_size=None):
    if model_type == "unet":
        from monai.networks.nets import UNet
        m = UNet(
            spatial_dims=3, in_channels=1, out_channels=1,
            channels=(64, 128, 256, 512, 1024), strides=(2, 2, 2, 2),
            num_res_units=2, norm="batch",
        )
    elif model_type == "swinunetr":
        from monai.networks.nets import SwinUNETR
        m = SwinUNETR(
            img_size=patch_size, in_channels=1, out_channels=1,
            feature_size=48, spatial_dims=3, depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24), drop_rate=0.1, attn_drop_rate=0.1,
            dropout_path_rate=0.3, normalize=True, use_checkpoint=True,
        )
    elif model_type == "segmamba":
        from model_segmamba.segmamba import SegMamba
        m = SegMamba(
            in_chans=1, out_chans=1, depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384], res_block=True, spatial_dims=3,
        )
    elif model_type == "flexmamba":
        from model_segmamba.flexiblemamba import FlexibleMamba
        depths = depths or [2, 2, 2, 2, 2]
        feat_size = feat_size or [48, 96, 192, 384, 768]
        m = FlexibleMamba(
            in_chans=1, 
            out_chans=1, 
            depths=depths,
            feat_size=feat_size, 
            hidden_size=768,
            res_block=True, 
            spatial_dims=3,
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")
    return m.cuda() if use_cuda else m


class sCTTrainer(Trainer):
    def __init__(self, model_type, model_path, output_dir, env_type,
                 max_epochs, batch_size, device="cpu", val_every=1,
                 num_gpus=1, logdir="./logs/", master_ip="localhost",
                 master_port=17750, training_script="train.py",
                 model_depths=None, model_feat_size=None):
        super().__init__(env_type, max_epochs, batch_size, device, val_every,
                         num_gpus, logdir, master_ip, master_port, training_script)
        self.patch_size   = patch_size
        self.augmentation = False
        self.model_path   = model_path
        self.save_path    = output_dir
        self.model_depths = model_depths
        self.model_feat_size = model_feat_size
        os.makedirs(self.save_path, exist_ok=True)

        self.model, self.window_infer, self.predictor = self._load_model(model_type)
        print(f"[INFO] Output directory: {self.save_path}")

    # ------------------------------------------------------------------
    def get_input(self, batch):
        return batch["data"], batch["seg"]

    def post_process(self, pred: np.ndarray) -> np.ndarray:
        """
        /1000-normalised model output → HU integers.
        Training used CT/1000, so inverse is *1000 then clip to [-1024, 3000].
        """
        pred_hu = pred.astype(np.float32) * 1000.0
        pred_hu = np.clip(pred_hu, -1024.0, 3000.0)
        return np.round(pred_hu).astype(np.int32)

    def compute_metrics(self, pred_hu: np.ndarray, target_hu: np.ndarray) -> dict:
        """MAE, MSE, PSNR, SSIM — all in HU space."""
        a = pred_hu.astype(np.float32)
        b = target_hu.astype(np.float32)
        data_range = float(np.max(b) - np.min(b))

        mae  = float(np.mean(np.abs(a - b)))
        mse  = float(np.mean((a - b) ** 2))
        psnr = (20.0 * math.log10(data_range) - 10.0 * math.log10(mse)) if mse > 0 else float("inf")

        ssim_loss = SSIMLoss(spatial_dims=3, data_range=max(data_range, 1.0))
        ssim_val  = ssim_loss(
            torch.from_numpy(a).unsqueeze(0).unsqueeze(0),
            torch.from_numpy(b).unsqueeze(0).unsqueeze(0),
        ).item()
        return {"mae": mae, "mse": mse, "psnr": psnr, "ssim": ssim_val}

    # ------------------------------------------------------------------
    def _load_model(self, model_type):
        self.device = device
        model       = build_model(model_type, use_cuda="cuda" in device,
                                 depths=self.model_depths, feat_size=self.model_feat_size)

        raw = torch.load(self.model_path, map_location="cpu", weights_only=False)
        # unwrap DDP / torch.compile wrappers
        sd  = raw.get("module", raw)
        new_sd = {}
        for k, v in sd.items():
            k = k.removeprefix("module.")    # DDP
            k = k.removeprefix("_orig_mod.") # torch.compile
            new_sd[k] = v
        model.load_state_dict(new_sd, strict=True)
        model.eval()
        print(f"[INFO] Loaded checkpoint: {self.model_path}")

        window_infer = SlidingWindowInferer(
            roi_size=patch_size, sw_batch_size=1, overlap=0.5,
            progress=False, mode="gaussian",
        )
        predictor = Predictor(window_infer=window_infer, mirror_axes=[0, 1, 2])
        self.window_infer = window_infer
        self.predictor    = predictor
        return model, window_infer, predictor

    # ------------------------------------------------------------------
    def validation_step(self, batch, save_predictions=True):
        image, target = self.get_input(batch)
        properties    = batch.get("properties", {})

        # case name — dataloader wraps each field in a list for batch dim
        raw_name  = properties.get("name", None)
        case_name = raw_name[0] if isinstance(raw_name, (list, tuple)) else str(raw_name)
        if not case_name:
            case_name = f"case_{int(torch.randint(0, int(1e9), (1,)).item())}"
            print(f"[WARNING] No case name in properties — using: {case_name}")

        # Spacing: stored in SimpleITK convention (x, y, z)
        raw_spacing = properties.get("spacing", None)
        if raw_spacing is not None:
            # unwrap batch-list wrapping if present
            if isinstance(raw_spacing, (list, tuple)) and len(raw_spacing) == 1:
                raw_spacing = raw_spacing[0]
            spacing_itk = tuple(float(s) for s in raw_spacing)
        else:
            spacing_itk = (1.0, 1.0, 3.0)
            print(f"[WARNING] spacing not in properties — using default {spacing_itk}")

        # Sliding-window inference
        self.model.eval()
        with torch.no_grad():
            image = image.to(self.device)
            t0    = time.time()
            pred  = self.window_infer(image, self.model)
        elapsed = time.time() - t0
        h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
        print(f"[{case_name}] Inference in {h:02d}:{m:02d}:{s:02d}")

        pred_arr   = pred.squeeze().cpu().numpy()       # /1000-normalised, float32
        target_arr = target.squeeze().cpu().numpy()     # /1000-normalised, float32

        # Post-process: *1000, clip [-1024, 3000]
        pred_hu   = self.post_process(pred_arr)
        target_hu = np.clip(target_arr * 1000.0, -1024.0, 3000.0).astype(np.float32)

        metrics = self.compute_metrics(pred_hu.astype(np.float32), target_hu)
        print(f"[{case_name}] MAE={metrics['mae']:.2f} HU  "
              f"PSNR={metrics['psnr']:.2f} dB  SSIM={metrics['ssim']:.4f}")

        if save_predictions:
            out_path = os.path.join(self.save_path, f"{case_name}.nii.gz")
            pred_itk = sitk.GetImageFromArray(pred_hu.astype(np.float32))
            # Preserve image metadata when possible to avoid spatial misalignment
            # Prefer copying SimpleITK image if provided in properties
            sitk_image = properties.get("sitk_image", None)
            if sitk_image is not None and hasattr(sitk_image, "GetDirection"):
                try:
                    pred_itk.CopyInformation(sitk_image)
                except Exception:
                    pred_itk.SetSpacing(spacing_itk)
            else:
                # Fallback: set spacing and any origin/direction available in properties
                try:
                    pred_itk.SetSpacing(spacing_itk)
                except Exception:
                    pass
                origin = properties.get("origin", None)
                direction = properties.get("direction", None)
                if origin is not None:
                    try:
                        pred_itk.SetOrigin(tuple(origin))
                    except Exception:
                        pass
                if direction is not None:
                    try:
                        pred_itk.SetDirection(tuple(direction))
                    except Exception:
                        pass

            sitk.WriteImage(pred_itk, out_path)
            print(f"[{case_name}] Saved → {out_path}")

        return metrics.get("mae", float("nan"))

    # kept for API compatibility with base class
    def filte_state_dict(self, sd):
        sd = sd.get("module", sd)
        new_sd = {}
        for k, v in sd.items():
            k = k.removeprefix("module.")
            k = k.removeprefix("_orig_mod.")
            new_sd[k] = v
        return new_sd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference MRI→CT synthesis model.")
    parser.add_argument("--model_type", required=True, choices=["unet", "swinunetr", "segmamba", "flexmamba"])
    parser.add_argument("--model_path", required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--input_dir",
                        default=os.path.join(os.environ.get("WORKSPACE", "/workspace"),
                                             "SegMamba_Translation/preprocessed/fullres/train"),
                        help="Directory containing preprocessed .npz/.npy/.pkl files")
    parser.add_argument("--output_dir",
                        default=os.path.join(os.environ.get("WORKSPACE", "/workspace"),
                                             "SegMamba_Translation/results"),
                        help="Directory where predicted .nii.gz files will be saved")
    parser.add_argument("--use_val_split", action="store_true",
                        help="Run on the 10%% val split (same seed as training) instead of all data")
    parser.add_argument("--model_depths", default="2,2,2,2",
                        help="Comma-separated Mamba depths for FlexibleMamba.")
    parser.add_argument("--model_feat_size", default="48,96,192,384",
                        help="Comma-separated feature sizes for FlexibleMamba.")
    parser.add_argument("--anatomical_region", default="all", choices=["all", "AB", "HN", "TH"],
                        help="Anatomical-region ablation. Use all, AB, HN, or TH. Default: all.")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    print(f"model_type     : {args.model_type}")
    print(f"model_path     : {args.model_path}")
    print(f"input_dir      : {args.input_dir}")
    print(f"output_dir     : {args.output_dir}")
    print(f"use_val_split  : {args.use_val_split}")
    print(f"anatomical_region : {args.anatomical_region}")

    model_depths = [int(x) for x in args.model_depths.split(",") if x.strip()]
    model_feat_size = [int(x) for x in args.model_feat_size.split(",") if x.strip()]

    trainer = sCTTrainer(
        model_type=args.model_type,
        model_path=args.model_path,
        output_dir=args.output_dir,
        env_type=env,
        max_epochs=max_epoch,
        batch_size=batch_size,
        device=device,
        logdir="",
        val_every=val_every,
        num_gpus=num_gpus,
        master_port=17751,
        training_script=__file__,
        model_depths=model_depths,
        model_feat_size=model_feat_size,
    )

    if args.use_val_split:
        # 10% val split — identical to the split used during training
        _, val_ds, _ = get_train_val_test_loader_from_train(
            args.input_dir, train_rate=0.9, val_rate=0.1, test_rate=0.01, seed=42,
        )
        infer_ds = val_ds
    else:
        # All cases in input_dir
        _, _, infer_ds = get_train_val_test_loader_from_train(
            args.input_dir, train_rate=0.01, val_rate=0.0, test_rate=1.0, seed=42, anatomical_region=args.anatomical_region
        )

    athens_time = datetime.now(ZoneInfo("Europe/Athens"))
    print(f"Starting predictions: {athens_time.strftime('%d/%m/%Y %H:%M')} (Athens)")
    trainer.validation_single_gpu(infer_ds, save_predictions=True)
