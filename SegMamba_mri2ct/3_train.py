import argparse
import os
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
from light_training.trainer import Trainer
from light_training.utils.files_helper import save_new_model_and_delete_last
from monai.inferers import SlidingWindowInferer
from monai.losses import SSIMLoss
from monai.utils import set_determinism

set_determinism(1234)


MODEL_DEFAULTS = {
    "unet":      {"max_epoch": 500, "val_every": 50, "save_subdir": "baseline_unet"},
    "swinunetr": {"max_epoch": 500, "val_every": 50, "save_subdir": "baseline_swinunetr"},
    "segmamba":  {"max_epoch": 500, "val_every": 50, "save_subdir": "baseline_segmamba"},
}

# Training hyper-parameters 
env           = "pytorch"
batch_size    = 2
num_gpus      = 1
device        = "cuda:0"
roi_size      = [64, 192, 192]
# augmentation = "nomirror"
augmentation  = False
switch_to_compound_loss = 99   # epoch at which SSIM + AFP terms are added


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


# ── AFP perceptual loss ───────────────────────────────────────────────────────
class AFP(nn.Module):
    def __init__(self, net: str = "", layers=[], normalize_before_L1=False):
        super().__init__()
        module_dir = os.path.dirname(os.path.abspath(__file__))
        model_params = {
            "TotalSeg117": {
                "weights_path": os.path.join(module_dir, "checkpoint_6mm_final.pth"),
                "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                "kernels": [[3, 3, 3]] * 5,
                "num_classes": 118,
                "model_type": "PlainConvUNet_5",
            },
            "TotalSeg_ABHNTH_117labels": {  # 1×1×3 mm, RIKEN
                "weights_path": os.path.join(module_dir, "checkpoint_final.pth"),
                "strides": [[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
                "kernels": [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                "num_classes": 118,
                "model_type": "PlainConvUNet",
            },
        }
        params = model_params[net]
        if params["model_type"] == "PlainConvUNet":
            self.layers = layers if layers else [0, 1, 2, 3, 4, 5, 6, 7]
            self.stages = 5
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
        elif params["model_type"] == "PlainConvUNet_5":
            self.layers = layers if layers else [0, 1, 2, 3, 4, 5, 6]
            self.stages = 4
            model = PlainConvUNet(
                input_channels=1, n_stages=5,
                features_per_stage=[32, 64, 128, 256, 320],
                conv_op=nn.Conv3d, kernel_sizes=[[3, 3, 3]] * 5,
                strides=params["strides"], num_classes=params["num_classes"],
                deep_supervision=False, n_conv_per_stage=[2] * 5,
                n_conv_per_stage_decoder=[2] * 4, conv_bias=True,
                norm_op=nn.InstanceNorm3d, norm_op_kwargs={"eps": 1e-5, "affine": True},
                nonlin=nn.LeakyReLU, nonlin_kwargs={"inplace": True},
            )

        if not os.path.exists(params["weights_path"]):
            raise FileNotFoundError(f"AFP checkpoint not found: {params['weights_path']}")
        checkpoint       = torch.load(params["weights_path"], map_location="cuda", weights_only=False)
        model_state_dict = checkpoint.get("state_dict", checkpoint.get("network_weights", checkpoint.get("model_state_dict")))
        model.load_state_dict(model_state_dict, strict=True)
        print(f"AFP, layers {layers}, loaded {net}: {params['weights_path']}")
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
        self.model = model.to(device="cuda", dtype=torch.float16)

        self.L1                      = nn.L1Loss()
        self.net                     = net
        self.print_perceptual_layers = False
        self.print_loss              = False
        self.normalize_before_L1     = normalize_before_L1

    def center_pad_to_multiple_of_2pow(self, x):
        factor = 2 ** self.stages
        pad    = []
        for s in reversed(x.shape[-3:]):
            new   = ((s + factor - 1) // factor) * factor
            total = new - s
            pad.extend([total // 2, total - total // 2])
        return F.pad(x, pad, mode="constant", value=0)

    def forward(self, x, y):

        x = self.center_pad_to_multiple_of_2pow(x)
        y = self.center_pad_to_multiple_of_2pow(y)

        with torch.no_grad():
            emb_x = self.model(x)
            emb_y = self.model(y)
        self.emb_x = emb_x
        self.emb_y = emb_y

        AFP_loss = 0
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
    def __init__(self, net="", w_afp=0.04, w_ssim=0.7, w_l1=1.0, w_bone_l1=1.0,
                 z_score_mean=-219.1395, z_score_std=414.3167):
        super().__init__()
        self.z_score_mean = z_score_mean
        self.z_score_std  = z_score_std

        # Precompute z-score equivalents of HU clip bounds and tissue thresholds
        self.clip_min_z = (-1024.0 - z_score_mean) / z_score_std
        self.clip_max_z = ( 1500.0 - z_score_mean) / z_score_std
        self.bone_z     = (  300.0 - z_score_mean) / z_score_std
        self.air_z      = ( -700.0 - z_score_mean) / z_score_std

        # SSIM data_range in z-score space = (HU range) / std
        self.ssim_loss  = SSIMLoss(spatial_dims=3, data_range=self.clip_max_z - self.clip_min_z)
        self.l1_loss    = nn.L1Loss()
        self.afp        = AFP(net="TotalSeg_ABHNTH_117labels", normalize_before_L1=True)
        self.w_l1       = w_l1
        self.w_bone_l1  = w_bone_l1
        self.w_afp      = w_afp
        self.w_ssim     = w_ssim
        self.counter    = 0

    def bone_weighted_l1(self, pred, target, w_bone=3.0, w_soft=1.5, w_air=0.5):
        # Pred and target are in z-score space; thresholds are precomputed z-score equivalents of HU boundaries
        target    = torch.clip(target, self.clip_min_z, self.clip_max_z)
        pred      = torch.clip(pred,   self.clip_min_z, self.clip_max_z)
        bone_mask = target >  self.bone_z
        soft_mask = (target <= self.bone_z) & (target > self.air_z)
        air_mask  = target <= self.air_z
        weights   = (w_bone * bone_mask + w_soft * soft_mask + w_air * air_mask).float()
        return torch.mean(weights * torch.abs(pred - target)) / (weights.mean() + 1e-6)

    def forward(self, x, y, current_epoch=0):
        # x and y are z-score preprocessed; all losses operate directly in z-score space
        l1      = self.l1_loss(x, y)
        bone_l1 = self.bone_weighted_l1(x, y)
        if current_epoch > switch_to_compound_loss:
            ssim = self.ssim_loss(x, y)
            with torch.amp.autocast("cuda"):
                afp_loss = self.afp(x, y)
            loss = (self.w_l1 * l1 + self.w_bone_l1 * bone_l1 + self.w_ssim * ssim + self.w_afp * afp_loss) / 2.0
            if self.counter % 100 == 0:
                print(f"[LOSS] L1={l1.item():.4f}  SSIM={ssim.item():.4f}  AFP={afp_loss.item():.4f}  Bone={bone_l1.item():.4f}")
        else:
            loss = self.w_l1 * l1 + self.w_bone_l1 * bone_l1
            if self.counter % 100 == 0:
                print(f"[LOSS] L1={l1.item():.4f}  Bone={bone_l1.item():.4f}")
        self.counter += 1
        return loss


# ── Trainer ───────────────────────────────────────────────────────────────────
class sCTTrainer(Trainer):
    def __init__(self, model_type, model_save_path, env_type, max_epochs, batch_size,
                 device="cpu", val_every=1, num_gpus=1, logdir="./logs/",
                 master_ip="localhost", master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus,
                         logdir, master_ip, master_port, training_script)
        self.window_infer        = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=1,
                                                        overlap=0.5, mode="gaussian")
        self.augmentation        = augmentation
        self.device              = device
        self.model               = build_model(model_type, use_cuda="cuda" in device)
        self.patch_size          = roi_size
        self.num_patches_per_batch = 1
        self.best_mae            = float("inf")
        self.print_time          = False
        self.train_process       = 8
        self.val_step            = 0
        self.model_save_path     = model_save_path
        self.counter             = 0
        self.z_score_mean        = -219.1395
        self.z_score_std         = 414.3167

        self.l1_loss  = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.loss     = compound_loss(net="TotalSeg_ABHNTH_117labels")

        self.optimizer      = torch.optim.AdamW(self.model.parameters(), lr=5e-4, weight_decay=1e-5)
        self.scheduler_type = "poly"

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
        return self.loss(pred, target, self.epoch)

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

        output_hu  = output * self.z_score_std + self.z_score_mean
        target_hu  = target * self.z_score_std + self.z_score_mean
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
        print(f"Validation → MAE={mean_mae:.4f}  PSNR={mean_psnr:.2f}")

        if mean_mae < self.best_mae:
            self.best_mae = mean_mae
            save_new_model_and_delete_last(
                self.model,
                os.path.join(self.model_save_path, f"best_model_mae_{mean_mae:.4f}.pt"),
                delete_symbol="best_model",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MRI→CT synthesis model.")
    parser.add_argument("--model_type",      required=True, choices=["unet", "swinunetr", "segmamba"],
                        help="Model architecture")
    parser.add_argument("--data_dir",        default=os.path.join(os.environ.get("WORKSPACE", "/workspace"),
                                                                   "data/raw_data/fullres/train"))
    parser.add_argument("--logdir",          default=os.path.join(os.environ.get("WORKSPACE", "/workspace"),
                                                                   "logs/segmamba"))
    parser.add_argument("--model_save_path", default=None,
                        help="Checkpoint directory. Defaults to <logdir>/<model_type>")
    parser.add_argument("--max_epoch",       type=int, default=500)
    parser.add_argument("--val_every",       type=int, default=50)
    args = parser.parse_args()

    defaults        = MODEL_DEFAULTS[args.model_type]
    max_epoch       = args.max_epoch       or defaults["max_epoch"]
    val_every       = args.val_every       or defaults["val_every"]
    model_save_path = args.model_save_path or os.path.join(args.logdir, defaults["save_subdir"])

    os.makedirs(model_save_path, exist_ok=True)
    print(f"model_type      : {args.model_type}")
    print(f"data_dir        : {args.data_dir}")
    print(f"logdir          : {args.logdir}")
    print(f"model_save_path : {model_save_path}")
    print(f"max_epoch       : {max_epoch}  |  val_every: {val_every}")

    warnings.filterwarnings("ignore")

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
    )

    train_ds, val_ds, _ = get_train_val_test_loader_from_train(
        args.data_dir, train_rate=0.9, val_rate=0.1, test_rate=0.01, seed=42
    )
    athens_time = datetime.now(ZoneInfo("Europe/Athens"))
    print(f"Starting training: {athens_time.strftime('%d/%m/%Y %H:%M')} (Athens)")
    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
