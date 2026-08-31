import argparse
import os
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo
import wandb

# Must happen before the first CUDA API call (i.e. before any .cuda() / torch.cuda.*).
# The NVIDIA Container Runtime overwrites CUDA_VISIBLE_DEVICES at container start;
# passing TRAINING_GPU=5 via `docker run -e TRAINING_GPU=5` lets us restore the
# correct physical device ID here so MPS sees a consistent device on both sides.
_gpu_id = os.environ.get("TRAINING_GPU")
if _gpu_id is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_id
    print(f"[INIT] CUDA_VISIBLE_DEVICES set to {_gpu_id} (from TRAINING_GPU)")

# torch.compile (inductor) calls getpwuid() to build its cache path, which fails
# when running as --user UID:GID with a UID not present in /etc/passwd.
# Pre-setting TORCHINDUCTOR_CACHE_DIR bypasses the username lookup entirely.
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(
    os.environ.get("HOME", "/tmp"), "torchinductor_cache"
))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
from light_training.trainer import Trainer
from light_training.utils.files_helper import save_new_model_and_delete_last
from monai.inferers import SlidingWindowInferer
from monai.utils import set_determinism
from monai.metrics import compute_ms_ssim

WANDB_API_KEY = "ADD_TOKEN_HERE"
os.environ.setdefault("WANDB_API_KEY", WANDB_API_KEY)
try:
    wandb.login(key=WANDB_API_KEY, relogin=False)
except Exception as e:
    print(f"[WANDB] login failed: {e}")

set_determinism(1234)

MODEL_DEFAULTS = {
    "unet":      {"max_epoch": 1500, "val_every": 150, "save_subdir": "baseline_unet"},
    "swinunetr": {"max_epoch": 1500, "val_every": 150, "save_subdir": "baseline_swinunetr"},
    "segmamba":  {"max_epoch": 1500, "val_every": 150, "save_subdir": "baseline_segmamba"},
    "flexmamba": {"max_epoch": 1500, "val_every": 150, "save_subdir": "baseline_flexmamba"},
}

# Training hyper-parameters 
env           = "pytorch"
batch_size    = 2
num_gpus      = 1
device        = "cuda:0"
roi_size      = [64, 224, 224]
augmentation  = False
switch_to_compound_loss = 0    # 0 = use AFP from epoch 0


def build_model(model_type: str, use_cuda: bool, depths=None, feat_size=None):
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
            img_size=tuple(roi_size),
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
    elif model_type == "flexmamba":
        from model_segmamba.flexiblemamba import FlexibleMamba
        depths = depths or [2, 2, 2, 2]
        feat_size = feat_size or [48, 96, 192, 384]
        m = FlexibleMamba(
            in_chans=1,
            out_chans=1,
            depths=depths,
            feat_size=feat_size,
            hidden_size=768,
            norm_name = ("group", {"num_groups": 8}),
            res_block=True,
            spatial_dims=3,
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from: unet, swinunetr, segmamba, flexmamba")

    if not use_cuda:
        return m
    m = m.to("cuda", non_blocking=True)
    m = torch.compile(m)
    return m


# ── AFP perceptual loss ───────────────────────────────────────────────────────
# TotalSeg117 (Dataset297, 3mm fast): isotropic PlainConvUNet.
# Expected input: CTNormalization — clip HU to [−1004, 1588], then z-score.
# Intensity stats come from the checkpoint's foreground_intensity_properties.
AFP_SPACING      = (3.0, 3.0, 3.0)
PATCH_SPACING    = (3.0, 1.0, 1.0)
AFP_HU_CLIP_MIN  = -1004.0
AFP_HU_CLIP_MAX  =  1588.0
AFP_HU_MEAN      =   -50.387
AFP_HU_STD       =   503.392

class AFP(nn.Module):
    def __init__(self, patch_spacing=PATCH_SPACING, afp_spacing=AFP_SPACING,
                 layers=None, normalize_before_L1=False):
        super().__init__()
        module_dir   = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(module_dir, "checkpoint_3mm_fast.pth")

        self.layers = layers if layers is not None else [0, 1, 2, 3, 4, 5, 6]
        self.stages = 4
        model = PlainConvUNet(
            input_channels=1, n_stages=5,
            features_per_stage=[32, 64, 128, 256, 320],
            conv_op=nn.Conv3d, kernel_sizes=[[3, 3, 3]] * 5,
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            num_classes=118, deep_supervision=False,
            n_conv_per_stage=[2] * 5, n_conv_per_stage_decoder=[2] * 4,
            conv_bias=True, norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            nonlin=nn.LeakyReLU, nonlin_kwargs={"inplace": True},
        )

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"AFP checkpoint not found: {weights_path}")
        checkpoint       = torch.load(weights_path, map_location="cuda", weights_only=False)
        model_state_dict = checkpoint.get("state_dict", checkpoint.get("network_weights", checkpoint.get("model_state_dict")))
        model.load_state_dict(model_state_dict, strict=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        def _forward_with_features(x):
            skips      = model.encoder(x)
            decoder    = model.decoder
            lres_input = skips[-1]
            all_feature_maps = []
            for s in range(len(decoder.stages)):
                x = decoder.transpconvs[s](lres_input)
                x = torch.cat((x, skips[-(s + 2)]), 1)
                x = decoder.stages[s](x)
                all_feature_maps.append(x)
                if s == (len(decoder.stages) - 1):
                    all_feature_maps.append(decoder.seg_layers[-1](x))
                lres_input = x
            return skips[:-2] + all_feature_maps

        model.forward = _forward_with_features
        self.model = model.to(device="cuda", dtype=torch.float16, non_blocking=True)

        self.L1                      = nn.L1Loss()
        self.patch_spacing           = patch_spacing
        self.afp_spacing             = afp_spacing
        self.scale_factors           = tuple(p / a for p, a in zip(patch_spacing, afp_spacing))
        self.print_perceptual_layers = False
        self.print_loss              = False
        self.normalize_before_L1     = normalize_before_L1

    def center_pad_to_multiple_of_2pow(self, x):
        _PAD_VALUE = -1.9
        factor = 2 ** self.stages
        pad    = []
        for s in reversed(x.shape[-3:]):
            new   = ((s + factor - 1) // factor) * factor
            total = new - s
            pad.extend([total // 2, total - total // 2])
        return F.pad(x, pad, mode="constant", value=_PAD_VALUE)

    def _to_afp_space(self, x):
        x_hu   = torch.clamp(x * 1000.0, AFP_HU_CLIP_MIN, AFP_HU_CLIP_MAX)
        x_norm = (x_hu - AFP_HU_MEAN) / AFP_HU_STD
        if any(abs(s - 1.0) > 1e-3 for s in self.scale_factors):
            x_norm = F.interpolate(x_norm.float(), scale_factor=self.scale_factors,
                                   mode="trilinear", align_corners=False)
        return x_norm

    def forward(self, x, y):
        x = self.center_pad_to_multiple_of_2pow(self._to_afp_space(x))
        y = self.center_pad_to_multiple_of_2pow(self._to_afp_space(y))

        # with torch.no_grad():
        emb_x = self.model(x)
        emb_y = self.model(y)
        self.emb_x = emb_x
        self.emb_y = emb_y

        AFP_loss = 0.0
        for i in self.layers:
            if self.normalize_before_L1:
                emb_x[i] = F.instance_norm(emb_x[i])
                emb_y[i] = F.instance_norm(emb_y[i])
            layer_loss = self.L1(emb_x[i], emb_y[i].detach())
            AFP_loss  += layer_loss
            if self.print_perceptual_layers:
                print(f"Layer {i}, {emb_x[i].shape} | L1: {layer_loss.item():.4f}")
        if self.print_loss:
            print(f"AFP_total: {AFP_loss:.5f}")
        return AFP_loss


# ── Compound loss ─────────────────────────────────────────────────────────────
class compound_loss(nn.Module):
    def __init__(self, w_l1=1.0, w_afp=0.03, w_ms_ssim=0.4, w_psnr=0.0, ms_ssim_weights=(0.3, 0.3, 0.4)):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.afp     = AFP()
        self.w_l1    = w_l1
        self.w_afp   = w_afp
        self.w_ms_ssim = w_ms_ssim
        self.w_psnr  = w_psnr
        self.ms_ssim_weights = ms_ssim_weights
        self.counter = 0

    def forward(self, x, y, current_epoch=0):
        l1 = self.l1_loss(x, y)
        data_range = (y.detach().max() - y.detach().min()).clamp(min=1e-6)
        ms_ssim_val = compute_ms_ssim(
            y_pred=x,
            y=y,
            spatial_dims=3,
            data_range=data_range,
            weights=self.ms_ssim_weights,
        )
        # Ensure ms_ssim_val is a scalar (compute_ms_ssim returns per-batch values)
        if ms_ssim_val.numel() > 1:
            ms_ssim_val = ms_ssim_val.mean()
        ms_ssim_loss = 1.0 - ms_ssim_val

        with torch.amp.autocast("cuda"):
            afp_loss = self.afp(x, y)

        loss = self.w_l1 * l1 + self.w_afp * afp_loss + self.w_ms_ssim * ms_ssim_loss

        psnr_val = None
        if self.w_psnr > 0.0:
            mse        = F.mse_loss(x, y)
            data_range = (y.detach().max() - y.detach().min()).clamp(min=1e-6)
            psnr_val   = 10.0 * torch.log10((data_range ** 2) / mse.clamp(min=1e-8))
            psnr_loss  = 1.0 / psnr_val.clamp(min=1e-6)
            loss      += self.w_psnr * psnr_loss


        self.last_components = {
            "mae": l1.detach().item(),
            "afp": afp_loss.detach().item(),
            "ms_ssim": ms_ssim_val.detach().item(),
            "ms_ssim_loss": ms_ssim_loss.detach().item(),
            "compound_loss": loss.detach().item(),
        }

        if psnr_val is not None:
            self.last_components["psnr"] = psnr_val.detach().item()    
        if self.counter % 100 == 0:
            print(f"[LOSS] L1={l1.item():.4f}  AFP={afp_loss.item():.4f}  MS-SSIM={ms_ssim_val.item():.4f}")
        self.counter += 1
        return loss


# ── Trainer ───────────────────────────────────────────────────────────────────
class sCTTrainer(Trainer):
    def __init__(self, model_type, model_save_path, env_type, max_epochs, batch_size,
                 device="cpu", val_every=1, num_gpus=1, logdir="./logs/",
                 master_ip="localhost", master_port=17750, training_script="train.py",
                 model_depths=None, model_feat_size=None,
                 w_afp=0.03, w_ms_ssim=0.4, ms_ssim_weights=(0.3, 0.3, 0.4),
                 lr: float = 5e-4, weight_decay: float = 1e-5, scheduler_type: str = "poly"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus,
                         logdir, master_ip, master_port, training_script)
        self.window_infer        = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=1,
                                                        overlap=0.5, mode="gaussian")
        self.augmentation        = augmentation
        self.device              = device
        self.model               = build_model(model_type, use_cuda="cuda" in device,
                                               depths=model_depths, feat_size=model_feat_size)
        self.patch_size          = roi_size
        self.num_patches_per_batch = 1
        self.best_mae            = float("inf")
        self.print_time          = False
        self.train_process       = 8
        self.val_step            = 0
        self.model_save_path     = model_save_path
        self.counter             = 0
        self.resume_epoch = 0
        self.epoch_compound_losses = []
        self._compound_loss_epoch = 0

        self.l1_loss  = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.loss     = compound_loss(w_l1=1.0, w_afp=w_afp, w_ms_ssim=w_ms_ssim,
                                      ms_ssim_weights=ms_ssim_weights)

        # Store optimizer / scheduler hyperparameters so they can be configured
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer      = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler_type = scheduler_type

        print(f"Current GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] Trainer initialised — model: {model_type}  save_path: {model_save_path}")
        print(self.model)

    def get_input(self, batch):
        image  = batch["data"]
        target = batch["seg"]
        if self.global_step < 20 and self.local_rank == 0:
            print(f"[DEBUG] Step {self.global_step}  CT dtype={target.dtype}  mean={target.float().mean().item():.4f}")
        return image, target

    def training_step(self, batch):
        image, target = self.get_input(batch)
        if self.global_step < 20 and self.local_rank == 0:
            print(f"Image device: {image.device}  Target device: {target.device}")
            print(f"Model device: {next(self.model.parameters()).device}")
        pred = self.model(image)
        if self.global_step < 20 and self.local_rank == 0:
            print(f"[DEBUG TRAIN] Step {self.global_step}  pred min={pred.float().min().item():.4f}  max={pred.float().max().item():.4f}")
        loss = self.loss(pred, target, self.epoch)

        if self.local_rank == 0:
            comps = self.loss.last_components

            if self.epoch != self._compound_loss_epoch and len(self.epoch_compound_losses) > 0:
                wandb.log({
                    "epoch": self._compound_loss_epoch,
                    "train_epoch/compound_loss": float(np.mean(self.epoch_compound_losses)),
                })
                self.epoch_compound_losses = []
                self._compound_loss_epoch = self.epoch

            wandb.log({
                "training_step": self.global_step,
                "train/mae": comps["mae"],
                "train/afp": comps["afp"],
                "train/ms_ssim": comps["ms_ssim"],
            })

            self.epoch_compound_losses.append(comps["compound_loss"])

        return loss

    def validation_step(self, batch):
        image, target = self.get_input(batch)
        if self.epoch < 100 and self.local_rank == 0:
            print(f"[DEBUG VAL] target={target.shape}  image={image.shape}"
                  f"  CT mean={target.float().mean().item():.4f}"
                  f"  min={target.float().min().item():.4f}  max={target.float().max().item():.4f}")
        self.model.eval()
        with torch.no_grad():
            output = self.model(image)
        if self.val_step < 20 and self.local_rank == 0:
            print(f"[DEBUG VAL] output min={output.float().min().item():.4f}"
                  f"  max={output.float().max().item():.4f}"
                  f"  mean={output.float().mean().item():.4f}")

        output_hu  = output * 1000.0
        target_hu  = target * 1000.0
        mae        = self.l1_loss(output_hu, target_hu).item()
        data_range = max(target_hu.float().max().item() - target_hu.float().min().item(), 1.0)
        mse        = self.mse_loss(output_hu, target_hu).item()
        psnr       = 10 * np.log10((data_range ** 2) / mse) if mse > 0 else 100.0
        print(f"[VAL] MAE={mae:.4f}  PSNR={psnr:.2f} dB")
        self.val_step += 1
        return [mae, psnr]

    def validation_end(self, val_outputs):
        if isinstance(val_outputs, list) and len(val_outputs) > 0:
            if isinstance(val_outputs[0], torch.Tensor):
                maes  = val_outputs[0].cpu().numpy()
                psnrs = val_outputs[1].cpu().numpy()
            else:
                maes  = [x[0] for x in val_outputs]
                psnrs = [x[1] for x in val_outputs]
        else:
            maes  = val_outputs.cpu().numpy() if hasattr(val_outputs, "cpu") else val_outputs
            psnrs = np.array([0])
            print("Assigned dummy value to PSNR")

        mean_mae  = np.mean(maes)
        mean_psnr = np.mean(psnrs)
        if self.local_rank == 0:
            wandb.log({
                "epoch": self.epoch,
                "val/mean_mae": mean_mae,
                "val/mean_psnr": mean_psnr,
            })
        print(f"Validation → MAE={mean_mae:.4f}  PSNR={mean_psnr:.2f}")

        if mean_mae < self.best_mae:
            self.best_mae = mean_mae
            save_new_model_and_delete_last(
                self.model,
                os.path.join(self.model_save_path, f"best_model_mae_{mean_mae:.4f}.pt"),
                delete_symbol="best_model",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MRI→CT synthesis model with FlexibleMamba.")
    parser.add_argument("--model_type",      required=True, choices=["unet", "swinunetr", "segmamba", "flexmamba"],
                        help="Model architecture")
    parser.add_argument("--data_dir",        default=os.path.join(os.environ.get("WORKSPACE", "/workspace"),
                                                                   "data/raw_data/fullres/train"))
    parser.add_argument(
        "--val_dir",
        default=os.path.join(os.environ.get("WORKSPACE", "/workspace"),
                                                                   "data/raw_data/fullres/excluded_data"))
    parser.add_argument("--logdir",          default=os.path.join(os.environ.get("WORKSPACE", "/workspace"),
                                                                   "logs/segmamba"))
    parser.add_argument("--model_save_path", default=None,
                        help="Checkpoint directory. Defaults to <logdir>/<model_type>")
    parser.add_argument("--max_epoch",       type=int, default=500)
    parser.add_argument("--val_every",       type=int, default=50)
    parser.add_argument("--model_depths",    default="2,2,2,2",
                        help="Comma-separated Mamba depths for FlexibleMamba.")
    parser.add_argument("--model_feat_size", default="48,96,192,384",
                        help="Comma-separated feature sizes for FlexibleMamba.")
    parser.add_argument("--w_afp", type=float, default=0.02,
                        help="Weight for AFP perceptual loss")
    parser.add_argument("--w_ms_ssim", type=float, default=0.3,
                        help="Weight for MS-SSIM loss term")
    parser.add_argument("--ms_ssim_weights", default="0.3,0.3,0.4",
                        help="Comma-separated MS-SSIM weights for 3 scales")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay for the optimizer")
    parser.add_argument("--scheduler_type", default="poly",
                        help="Scheduler type (default: 'poly')")
    parser.add_argument("--resume_epoch", type=int, default=0,
                        help="If >0, resume training starting from this epoch (int).")
    parser.add_argument("--resume_checkpoint", default=None,
                        help="Path to checkpoint file to load. If omitted and --resume_epoch>0, will look for checkpoint_ep<epoch-1>.pt in model_save_path")
    parser.add_argument("--anatomical_region", default="all", choices=["all", "AB", "HN", "TH"],
                        help="Anatomical-region ablation. Use all, AB, HN, or TH. Default: all.")
    args = parser.parse_args()

    defaults        = MODEL_DEFAULTS[args.model_type]
    max_epoch       = args.max_epoch       or defaults["max_epoch"]
    val_every       = args.val_every       or defaults["val_every"]
    model_save_path = args.model_save_path or os.path.join(args.logdir, defaults["save_subdir"])

    model_depths = [int(x) for x in args.model_depths.split(",") if x.strip()]
    model_feat_size = [int(x) for x in args.model_feat_size.split(",") if x.strip()]
    ms_ssim_weights = tuple(float(x) for x in args.ms_ssim_weights.split(",") if x.strip())

    lr = args.lr
    weight_decay = args.weight_decay
    scheduler_type = args.scheduler_type

    os.makedirs(model_save_path, exist_ok=True)
    print(f"model_type      : {args.model_type}")
    print(f"data_dir        : {args.data_dir}")
    print(f"logdir          : {args.logdir}")
    print(f"model_save_path : {model_save_path}")
    print(f"max_epoch       : {max_epoch}  |  val_every: {val_every}")
    print(f"anatomical_region : {args.anatomical_region}")

    # Print model characteristics and loss function's weights 
    if args.model_type == "flexmamba":
        print(f"model_depths    : {model_depths}")
        print(f"model_feat_size : {model_feat_size}")
        print(f"w_afp           : {args.w_afp}")
        print(f"w_ms_ssim       : {args.w_ms_ssim}")
        print(f"ms_ssim_weights : {ms_ssim_weights}")

    warnings.filterwarnings("ignore")

    # Set up wandb logging
    run_name = (
    f"{args.model_type}_"
    f"{datetime.now(ZoneInfo('Europe/Athens')).strftime('%Y%m%d_%H%M%S')}"
    )

    wandb.init(
        project="mri2ct",
        entity="ADD_ENTITY_HERE",
        name=run_name,
        config={
            "model_type": args.model_type,
            "batch_size": batch_size,
            "roi_size": roi_size,
            "max_epoch": max_epoch,
            "val_every": val_every,
            "lr": lr,
            "weight_decay": weight_decay,
            "w_afp": args.w_afp,
            "w_ms_ssim": args.w_ms_ssim,
            "ms_ssim_weights": ms_ssim_weights,
            "augmentation": augmentation,
            "anatomical_region": args.anatomical_region,
        },
    )

    wandb.define_metric("training_step")
    wandb.define_metric("train/mae", step_metric="training_step")
    wandb.define_metric("train/afp", step_metric="training_step")
    wandb.define_metric("train/ms_ssim", step_metric="training_step")

    wandb.define_metric("epoch")
    wandb.define_metric("train_epoch/compound_loss", step_metric="epoch")
    wandb.define_metric("val/mean_mae", step_metric="epoch")
    wandb.define_metric("val/mean_psnr", step_metric="epoch")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    trainer = sCTTrainer(
        model_type=args.model_type,
        model_save_path=model_save_path,
        env_type=env,
        max_epochs=max_epoch,
        batch_size=batch_size,
        device=device,
        logdir=args.logdir,
        val_every=val_every,
        num_gpus=num_gpus,
        master_port=17759,
        training_script=__file__,
        model_depths=model_depths,
        model_feat_size=model_feat_size,
        w_afp=args.w_afp,
        w_ms_ssim=args.w_ms_ssim,
        ms_ssim_weights=ms_ssim_weights,
        lr=lr,
        weight_decay=weight_decay,
        scheduler_type=scheduler_type,
    )

    train_ds, _, _ = get_train_val_test_loader_from_train(
        args.data_dir, train_rate=1.0, val_rate=0.0, test_rate=0.0, seed=39, anatomical_region=args.anatomical_region
    )
    _, val_ds, _ = get_train_val_test_loader_from_train(
        args.val_dir, train_rate=0.0, val_rate=1.0, test_rate=0.0, seed=39, anatomical_region=args.anatomical_region
    )
    # If requested, load checkpoint and optimizer state and set resume epoch
    if args.resume_epoch and args.resume_epoch > 0:
        trainer.resume_epoch = args.resume_epoch
        if args.resume_checkpoint:
            ckpt_path = args.resume_checkpoint
        else:
            # checkpoint files are saved as checkpoint_ep{epoch}.pt where epoch is 0-based
            ckpt_path = os.path.join(model_save_path, f"checkpoint_ep{trainer.resume_epoch-1}.pt")

        if os.path.exists(ckpt_path):
            try:
                sd = torch.load(ckpt_path, map_location="cpu")
                # load model weights (handles raw state_dict and wrapped dicts)
                trainer.load_state_dict(ckpt_path, strict=True)
                print(f"[RESUME] Model weights loaded from {ckpt_path}")
                # load optimizer state if present in checkpoint
                if isinstance(sd, dict) and "optimizer_state_dict" in sd and trainer.optimizer is not None:
                    try:
                        trainer.optimizer.load_state_dict(sd["optimizer_state_dict"])
                        print("[RESUME] Optimizer state loaded from checkpoint")
                    except Exception as e:
                        print(f"[RESUME] Failed to load optimizer state: {e}")
            except Exception as e:
                print(f"[RESUME] Failed to load checkpoint {ckpt_path}: {e}")
        else:
            print(f"[RESUME] Checkpoint not found: {ckpt_path}")
    _test_path = os.path.join(model_save_path, "write_test.pt")
    try:
        torch.save({"epoch": -1}, _test_path)
        os.remove(_test_path)
        print(f"[INIT] Write check passed: {model_save_path}")
    except Exception as e:
        raise RuntimeError(f"Cannot write to model_save_path '{model_save_path}': {e}") from e

    athens_time = datetime.now(ZoneInfo("Europe/Athens"))
    print(f"Starting training: {athens_time.strftime('%d/%m/%Y %H:%M')} (Athens)")
    try:
        trainer.train(train_dataset=train_ds, val_dataset=val_ds)
    finally:
        if wandb.run is not None:
            if len(trainer.epoch_compound_losses) > 0:
                wandb.log({
                    "epoch": trainer._compound_loss_epoch,
                    "train_epoch/compound_loss": float(np.mean(trainer.epoch_compound_losses)),
                })
            wandb.finish()