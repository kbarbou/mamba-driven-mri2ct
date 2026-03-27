from statistics import mode
import numpy as np
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
import torch 
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.trainer import Trainer
from monai.utils import set_determinism

import os
import argparse
from light_training.prediction import Predictor
import SimpleITK as sitk
import math
from monai.losses import SSIMLoss
from datetime import datetime
from zoneinfo import ZoneInfo
import time

set_determinism(123)

env = "pytorch"
max_epoch = 100
batch_size = 1
val_every = 10
num_gpus = 1
device = "cuda:0"
patch_size = [64, 192, 192]
debug_steps = 0

def build_model(model_type: str, use_cuda: bool):
    if model_type == "unet":
        from monai.networks.nets import UNet
        m = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(64, 128, 256, 512, 1024),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="batch",
        )
    elif model_type == "swinunetr":
        from monai.networks.nets import SwinUNETR
        m = SwinUNETR(
            img_size=(64, 192, 192),
            in_channels=1,
            out_channels=1,
            feature_size=48,
            spatial_dims=3,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            drop_rate=0.1,
            attn_drop_rate=0.1,
            dropout_path_rate=0.3,
            normalize=True,
            use_checkpoint=True,
        )
    elif model_type == "segmamba":
        from model_segmamba.segmamba import SegMamba
        m = SegMamba(
            in_chans=1,
            out_chans=1,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            res_block=True,
            spatial_dims=3,
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from: unet, swinunetr, segmamba")

    return m.cuda() if use_cuda else m

class sCTTrainer(Trainer):
    def __init__(self, model_type, model_path, output_dir, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        
        self.patch_size = patch_size
        self.augmentation = False

        self.model_path = model_path
        self.save_path = output_dir
        os.makedirs(self.save_path, exist_ok=True)

        self.model, self.predictor = self.define_model_segmamba(model_type)
        self.ssim_loss = SSIMLoss(spatial_dims=3, data_range=4095.0)

    def get_input(self, batch):
        image = batch["data"]
        target = batch["seg"]
    
        if self.global_step < debug_steps:
            if target is not None:
                try:
                    print(f"CT Data Type: {target.dtype}, MRI Data Type: {image.dtype}")
                    print(f"CT Data shape {target.shape}, MRI shape {image.shape}")
                    print(f"CT Mean: {target.float().mean().item()}, MRI Mean: {image.float().mean().item()}")
                except Exception:
                    pass

        #return image.float(), target.float()    
        return image, target 

    def post_process(self, pred, scaling="z_score"):
        if isinstance(pred, torch.Tensor):
            pred = pred.squeeze().cpu().numpy()

        if scaling == "min_max":
            pred = np.clip(pred, 0.0, 1.0)
            pred_hu = (pred * 2000.0) - 1000.0
            print(f"[INFO] inverse {scaling} and clipping applied in post_process.")
        elif scaling == "z_score":
            pred_hu = pred * 414.3167 - 219.1395 # Scale back to HU range
            pred_hu = np.clip(pred_hu, -1024.0, 3071.0)
            print(f"[INFO] inverse {scaling} and clipping applied in post_process.")
        else:
            pred_hu = np.clip(pred, -1024.0, 3071.0)
            print("[WARNING] Unknown scaling method, no scaling applied in post_process. (Only clipping)")

        # Convert to integer
        pred_hu = np.round(pred_hu).astype(np.int32)
        return pred_hu

    def compute_metrics(self, pred, target, mode="post_metrics"):
        """Compute MAE, MSE, PSNR and optional SSIM.

        mode=="pre": It is assumed that pred is normalized
        mode=="post": pred and target in HU.
        """
        # bring to numpy
        if isinstance(pred, torch.Tensor):
            pred_np = pred.squeeze().cpu().numpy()
        else:
            pred_np = np.array(pred)

        if isinstance(target, torch.Tensor):
            target_np = target.squeeze().cpu().numpy()
        else:
            target_np = np.array(target)
        
        assert pred_np.shape == target_np.shape, f"Shape mismatch between pred {pred_np.shape} and target {target_np.shape}."
        a = pred_np.astype(np.float32)
        b = target_np.astype(np.float32)
        maxv = float(np.max(b) - np.min(b))

        mae = float(np.mean(np.abs(a - b)))
        mse = float(np.mean((a - b) ** 2))
        if mse > 0:
            psnr = 20.0 * math.log10(maxv) - 10.0 * math.log10(mse)
        else:
            psnr = float('inf')
        if mode == "post_metrics":
            ssim_val = self.ssim_loss(torch.from_numpy(a).unsqueeze(0).unsqueeze(0), torch.from_numpy(b).unsqueeze(0).unsqueeze(0)).item()
        else:
            ssim_loss = SSIMLoss(spatial_dims=3, data_range=maxv)    
            ssim_val = ssim_loss(torch.from_numpy(a).unsqueeze(0).unsqueeze(0), torch.from_numpy(b).unsqueeze(0).unsqueeze(0)).item()

        return {"mae": mae, "mse": mse, "psnr": psnr, "ssim": ssim_val}

    def define_model_segmamba(self, model_type):
        self.device = device
        self.model = build_model(model_type, use_cuda="cuda" in device)                  
        print(self.model)
        new_sd = self.filte_state_dict(torch.load(self.model_path, map_location="cpu"))
        self.model.load_state_dict(new_sd)
        self.model.eval()
        window_infer = SlidingWindowInferer(roi_size=self.patch_size,
                                        sw_batch_size=1,
                                        overlap=0.25,
                                        progress=False,
                                        mode="constant")

        predictor = Predictor(window_infer=window_infer,
                              mirror_axes=[0,1,2])
        self.window_infer = window_infer
        self.predictor = predictor

        return self.model, predictor
    
    def validation_step(self, batch, save_predictions=False):
        #image, label, properties = self.get_input(batch)
        image, target = self.get_input(batch)
        properties = batch.get("properties", {})
        case_name = properties.get("name", None)
        if case_name is None:
            # Error
            case_name = f"case_{int(torch.randint(0,1e9,(1,)).item())}"
            print("[WARNING] Case name not found in properties, using random name:", case_name)
        

        # Apply Sliding window inference
        self.model.eval()
        with torch.no_grad():
            image = image.to(self.device) 
            target = target.to(self.device) if target is not None else None
            # expect img shape [B, C, D, H, W], SlidingWindowInferer expects batch dim
            try:
                if hasattr(self, "window_infer"):
                    start_time = time.time()
                    pred = self.window_infer(image, self.model)
                    elapsed = time.time() - start_time

                    hours = int(elapsed // 3600)
                    minutes = int((elapsed % 3600) // 60)
                    seconds = int(elapsed % 60)
                    print(f"Inference completed in {hours:02d}:{minutes:02d}:{seconds:02d} ")
                else:
                    pred = self.model(image)
            except Exception:
                # fallback: direct forward
                pred = self.model(image)

        # NOTE: Skip pre_metrics computation as it compares mismatched value ranges:
        # - pred_tensor is in normalized range (e.g., [-3, 3] for z-score)
        # - target is in HU range ([-1024, 3071])
        # This comparison is meaningless. Only post_metrics after denormalization is valid.
        pre_metrics = None

        pred_arr = pred.squeeze().cpu().numpy()
        pre_metrics = self.compute_metrics(pred_arr, target.squeeze(), mode="pre_metrics") if target is not None else None

        # Apply post-processing to get HU
        pred_hu = self.post_process(pred_arr, scaling="z_score")

        # Prepare output path
        out_path = os.path.join(self.save_path, f"{case_name[0]}.nii.gz")

        # Try to read original image to get spacing/origin/direction
        gt_folder = os.path.join("/workspace/raw_data/synthRAD2025/test/", case_name[0])
        if save_predictions:
            affine_ref = None
            if os.path.isdir(gt_folder):
                nii_files = [f for f in os.listdir(gt_folder) if f.endswith('.nii') or f.endswith('.nii.gz')]
                if len(nii_files) > 0:
                    ref_path = os.path.join(gt_folder, nii_files[1])
                    print("[INFO] Reading metadata from:", nii_files[1])
                    try:
                        ref_itk = sitk.ReadImage(ref_path)
                        pred_itk = sitk.GetImageFromArray(pred_hu.astype(np.float32))
                        pred_itk.SetSpacing(ref_itk.GetSpacing())
                        pred_itk.SetOrigin(ref_itk.GetOrigin())
                        pred_itk.SetDirection(ref_itk.GetDirection())
                        sitk.WriteImage(pred_itk, out_path)
                    except Exception:
                        # fallback to simple save
                        pred_itk = sitk.GetImageFromArray(pred_hu.astype(np.float32))
                        sitk.WriteImage(pred_itk, out_path)
                else:
                    pred_itk = sitk.GetImageFromArray(pred_hu.astype(np.float32))
                    sitk.WriteImage(pred_itk, out_path)
            else:
                pred_itk = sitk.GetImageFromArray(pred_hu.astype(np.float32))
                sitk.WriteImage(pred_itk, out_path)

        # Compute metrics against ground truth ct.nii.gz if available
        gt_path = os.path.join(gt_folder, "ct.nii.gz")

        post_metrics = None
        if os.path.exists(gt_path):
            try:
                gt_itk = sitk.ReadImage(gt_path)
                gt_arr = sitk.GetArrayFromImage(gt_itk).astype(np.float32)
                gt_arr = np.clip(gt_arr, -1024.0, 3071.0)
                # ensure same shape
                if gt_arr.shape != pred_hu.shape:
                    print(f"[DEBUG] : GT shape: {gt_arr.shape}, Pred shape: {pred_hu.shape}, Fixing mismatching via resampling.")
                    # try to resize pred to gt shape via simple interpolation
                    pred_resized = sitk.GetImageFromArray(pred_hu.astype(np.float32))
                    pred_resized.SetSpacing(ref_itk.GetSpacing())
                    pred_resized = sitk.Resample(pred_resized, gt_itk)
                    pred_arr = sitk.GetArrayFromImage(pred_resized)
                else:
                    pred_arr = pred_hu

                post_metrics = self.compute_metrics(pred_arr, gt_arr, mode="post_metrics")
            except Exception:
                post_metrics = None

        # print metrics summary
        pre_str = f"Pre_metrics: {pre_metrics}" if pre_metrics is not None else "pre_metrics: None"
        post_str = f"Post_metrics: {post_metrics}" if post_metrics is not None else "post_metrics: None"
        print(f"Saved prediction: {out_path}")
        print(pre_str)
        print(post_str)
        # return MAE (post) if available, else pre MAE, else nan
        if post_metrics is not None:
            return post_metrics.get("mae", float('nan'))
        if pre_metrics is not None:
            return pre_metrics.get("mae", float('nan'))
        return float('nan')
    
    def filte_state_dict(self, sd):
        if "module" in sd :
            sd = sd["module"]
        new_sd = {}
        for k, v in sd.items():
            k = str(k)
            new_k = k[7:] if k.startswith("module") else k 
            new_sd[new_k] = v 
        del sd 
        return new_sd
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference MRI→CT synthtesis model.")
    parser.add_argument("--model_type",      required=True, choices=["unet", "swinunetr", "segmamba"],
                        help="Model architecture")
    parser.add_argument("--model_path", default=None,
                        help="Checkpoint directory. Defaults to <logdir>/<model_type>")
    parser.add_argument("--input_dir",        default=os.path.join(os.environ.get("WORKSPACE", "/workspace"),
                                                                   "data/raw_data/fullres/test"))
    parser.add_argument("--output_dir",          default=os.path.join(os.environ.get("WORKSPACE", "/workspace"),
                                                                   "results/synthRAD2025/"))
    args = parser.parse_args()

    trainer = sCTTrainer(model_type=args.model_type,
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
                            training_script=__file__)
    
    _, _, test_ds = get_train_val_test_loader_from_train(args.input_dir, train_rate=0.01, val_rate=0.0, test_rate=1.00, seed=42)

    athens_time = datetime.now(ZoneInfo("Europe/Athens"))
    print(f"Starting predictions: {athens_time.strftime('%d/%m/%Y %H:%M')} (Timezone: Europe/Athens)")

    trainer.validation_single_gpu(test_ds, save_predictions=True)




