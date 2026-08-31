import torch 
import torch.nn as nn 
import os

from dynamic_network_architectures.architectures.unet import PlainConvUNet
import torch.nn.functional as F

from monai.losses import SSIMLoss
from monai.metrics import compute_ms_ssim

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