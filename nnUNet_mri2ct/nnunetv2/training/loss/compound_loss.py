import torch 
import torch.nn as nn 
import os

from dynamic_network_architectures.architectures.unet import PlainConvUNet
import torch.nn.functional as F

from monai.losses import SSIMLoss

class AFP(nn.Module):
    def __init__(self, net: str = "", layers=[], normalize_before_L1=False):
        super().__init__()
        module_dir = os.path.dirname(os.path.abspath(__file__))
        model_params = {
            "TotalSeg117": { #patch_size : [128 128 128], 0.6mm, 117 labels kept
                "weights_path": os.path.join(module_dir, "checkpoint_6mm_final.pth"),
                "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                "kernels" : [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                "num_classes": 118,
                "model_type": "PlainConvUNet_5"
            },
            "TotalSeg_ABHNTH_117labels": { #1*1*3mm, RIKEN
                "weights_path": os.path.join(module_dir, "checkpoint_final.pth"),
                "strides": [[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
                "kernels" : [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                "num_classes": 118,
                "model_type": "PlainConvUNet"
            },
        }
        params = model_params[net]
        if params["model_type"] == "PlainConvUNet":
            kernel = params.get("kernels", [[3, 3, 3]] * 6)
            self.layers = layers if layers else [0, 1, 2, 3, 4, 5, 6, 7]
            self.stages = 5
            model = PlainConvUNet(
                input_channels=1,
                n_stages=5,
                features_per_stage=[32, 64, 128, 256, 320],
                conv_op=nn.Conv3d,
                kernel_sizes=[[3, 3, 3]]*5,
                strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                num_classes=118,
                deep_supervision=False,
                n_conv_per_stage=[2]*5,
                n_conv_per_stage_decoder=[2]*4,
                conv_bias=True,
                norm_op=nn.InstanceNorm3d,
                norm_op_kwargs={'eps': 1e-5, 'affine': True},
                nonlin=nn.LeakyReLU,
                nonlin_kwargs={'inplace': True}
            )
        elif params["model_type"] == "PlainConvUNet_5":
            self.layers = layers if layers else [0, 1, 2, 3, 4, 5, 6]
            kernel = [[3, 3, 3]] * 5
            self.stages = 4
            model = PlainConvUNet(input_channels=1, n_stages=5, features_per_stage=[32, 64, 128, 256, 320],
                                conv_op=nn.Conv3d, kernel_sizes=kernel, strides=params["strides"],
                                num_classes=params["num_classes"], deep_supervision=False, n_conv_per_stage=[2] * 5,
                                n_conv_per_stage_decoder=[2] * 4, conv_bias=True, norm_op=nn.InstanceNorm3d,
                                norm_op_kwargs={'eps': 1e-5, 'affine': True}, nonlin=nn.LeakyReLU,
                                nonlin_kwargs={'inplace': True})
        
        if not os.path.exists(params["weights_path"]):
            raise FileNotFoundError(f'Error: Checkpoint not found at {params["weights_path"]}')
        checkpoint = torch.load(params["weights_path"], map_location='cuda', weights_only = False)
        model_state_dict = checkpoint.get('state_dict', checkpoint.get('network_weights', checkpoint.get('model_state_dict')))
        model.load_state_dict(model_state_dict, strict=True)
        print(f"AFP, layers {layers}, loaded {net} : {params['weights_path']}")
        model.eval()
  
        for param in model.parameters():
            param.requires_grad = False

        # Override forward to return intermediate feature maps (encoder skips + decoder features).
        # The pip-installed decoder only returns the final segmentation output,
        # but AFP needs per-layer features for perceptual loss computation.
        # This replicates the custom UNetDecoder.forward from nnUNet_translation.
        def _forward_with_features(x):
            skips = model.encoder(x)
            decoder = model.decoder
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

        self.model = model
        self.model = self.model.to(device='cuda', dtype=torch.float16) #arthur : needed for autocast ?

        self.L1 = nn.L1Loss()
        self.net = net
        self.print_perceptual_layers = False
        self.print_loss = False
        self.debug = False
        self.normalize_before_L1 = normalize_before_L1

    def center_pad_to_multiple_of_2pow(self, x):
        factor = 2 ** self.stages
        shape = x.shape[-3:]  
        pad = []
        for s in reversed(shape):  # reverse order for F.pad
            new = ((s + factor - 1) // factor) * factor
            total = new - s
            pad.extend([total // 2, total - total // 2])
        return F.pad(x, pad, mode='constant', value=0)
    
    def get_last_layer(self):
        return self.emb_x[-1], self.emb_y[-1]

    def forward(self, x, y): 
        """
        todo : check if normalization of input tensors is needed
        """
        x = self.center_pad_to_multiple_of_2pow(x)
        y = self.center_pad_to_multiple_of_2pow(y)

        emb_x = self.model(x)  
        emb_y = self.model(y)

        self.emb_x = emb_x
        self.emb_y = emb_y

        AFP_loss = 0
        layer_losses = []
        for i in self.layers:
            if self.normalize_before_L1:
                emb_x[i] = F.instance_norm(emb_x[i])
                emb_y[i] = F.instance_norm(emb_y[i])
            layer_loss = self.L1(emb_x[i], emb_y[i].detach())
            AFP_loss += layer_loss
            layer_losses.append((i, layer_loss.item()))

            if self.print_perceptual_layers:
                print(f"Layer {i}, {emb_x[i].shape} | L1: {layer_loss.item():.4f}")

        if self.print_loss:
            print(f"AFP_total: {AFP_loss:.5f}")
        return AFP_loss

class compound_loss(nn.Module):
    def __init__(self, net: str = "", w_afp=0.04, w_ssim=0.7, w_l1 = 1.0, w_bone_l1 = 1.0, z_score_mean = -219.1395, z_score_std = 414.3167, switch_to_compound_loss = 99):
        super().__init__()
        self.z_score_mean = z_score_mean
        self.z_score_std = z_score_std

        # Precompute z-score equivalents of the HU clip bounds and tissue thresholds
        self.clip_min_z = (-1024.0 - z_score_mean) / z_score_std   # ≈ -1.94 σ
        self.clip_max_z = (1500.0  - z_score_mean) / z_score_std   # ≈  4.15 σ
        self.bone_z     = (300.0   - z_score_mean) / z_score_std   # ≈  1.25 σ
        self.air_z      = (-700.0  - z_score_mean) / z_score_std   # ≈ -1.16 σ

        # SSIM data_range in z-score space = (HU range) / std
        data_range_z = self.clip_max_z - self.clip_min_z
        self.ssim_loss = SSIMLoss(spatial_dims=3, data_range=data_range_z)
        self.l1_loss = nn.L1Loss()
        self.afp = AFP(net="TotalSeg_ABHNTH_117labels", normalize_before_L1=True)
        self.w_l1 = w_l1
        self.w_bone_l1 = w_bone_l1
        self.w_afp = w_afp
        self.w_ssim = w_ssim
        self.counter = 0
        self.switch_to_compound_loss = switch_to_compound_loss

    def bone_weighted_l1(self, pred, target, w_bone=3.0, w_soft=1.5, w_air=0.5):
        # Pred and target are in z-score space; thresholds are precomputed z-score equivalents of HU boundaries

        target = torch.clip(target, self.clip_min_z, self.clip_max_z)
        pred   = torch.clip(pred,   self.clip_min_z, self.clip_max_z)

        bone_mask = (target > self.bone_z)
        soft_mask = (target <= self.bone_z) & (target > self.air_z)
        air_mask  = (target <= self.air_z)

        weights = (
            w_bone * bone_mask +
            w_soft * soft_mask +
            w_air  * air_mask
        ).float()

        l1 = torch.abs(pred - target)

        weighted_l1 = torch.mean(weights * l1) / (weights.mean() + 1e-6)

        return weighted_l1

    def forward(self, x, y, current_epoch=0):
        # x and y are z-score preprocessed; all losses operate directly in z-score space
        l1 = self.l1_loss(x, y)
        bone_l1 = self.bone_weighted_l1(x, y)
        if current_epoch > self.switch_to_compound_loss:
            ssim = self.ssim_loss(x, y)
            with torch.amp.autocast('cuda'):
                afp_loss = self.afp(x, y) 
            loss = (
                self.w_l1 * l1 +
                self.w_bone_l1 * bone_l1 +
                self.w_ssim * ssim + 
                self.w_afp * afp_loss 
            ) / 2.0
            if (self.counter % 100 == 0):
                print(f"[LOSS DEBUG] L1={l1.item():.4f}, "
                    f"SSIM={ssim.item():.4f}, "
                    f"AFP={afp_loss.item():.4f}, "
                    f"Bone={bone_l1.item():.4f}")
        else:
            loss = (
                self.w_l1 * l1 +
                self.w_bone_l1 * bone_l1
            )
            if (self.counter % 100 == 0):
                print(f"[LOSS DEBUG] L1={l1.item():.4f}, "
                    f"Bone={bone_l1.item():.4f}")
        self.counter += 1    
        return loss    