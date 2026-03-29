import torch
import numpy as np
from typing import Union, Tuple, List

from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import RenameTransform, NumpyToTensor
from batchgenerators.utilities.file_and_folder_operations import join
from time import time

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.training.loss.compound_loss import compound_loss
from nnunetv2.nets.UMambaEnc_3d import get_umamba_enc_3d_from_plans
from nnunetv2.nets.UMambaEnc_2d import get_umamba_enc_2d_from_plans
from torch import nn


class nnUNetTrainerUMambaEncNoAMP(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = False
        self.num_iterations_per_epoch = 250
        self.num_epochs = 1000
        self.batch_size = 1
        self.compound_loss = compound_loss(net="TotalSeg_ABHNTH_117labels")
        # Lower LR: nnUNet default 1e-2 + SGD momentum=0.99 is calibrated for segmentation
        # (softmax-bounded outputs). For unbounded z-score regression it diverges by epoch ~6.
        # AdamW with 3e-4 is the standard for regression/generation tasks.
        self.initial_lr = 3e-4
        self.weight_decay = 1e-5

    def configure_optimizers(self):
        # AdamW is more stable than SGD for unbounded regression targets (z-score CT generation).
        # SGD with lr=1e-2, momentum=0.99 can diverge on outlier batches even with grad clipping.
        from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    def _build_loss(self):
        return self.compound_loss

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) == 2:
            model = get_umamba_enc_2d_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels, deep_supervision=enable_deep_supervision)
        elif len(configuration_manager.patch_size) == 3:
            model = get_umamba_enc_3d_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels, deep_supervision=enable_deep_supervision)
        else:
            raise NotImplementedError("Only 2D and 3D models are supported")

        print("UMambaEnc: {}".format(model))

        return model

    @staticmethod
    def get_validation_transforms(
        deep_supervision_scales,
        is_cascaded: bool = False,
        foreground_labels=None,
        regions=None,
        ignore_label: int = None,
    ) -> AbstractTransform:
        # Skip RemoveLabelTransform: for CT generation the target is continuous float z-score,
        # not discrete integer labels. RemoveLabelTransform(-1, 0) would zero out all air tissue.
        val_transforms = [
            RenameTransform('seg', 'target', True),
            NumpyToTensor(['data', 'target'], 'float'),
        ]
        return Compose(val_transforms)

    @staticmethod
    def get_training_transforms(patch_size: Union[np.ndarray, Tuple[int]],
                                rotation_for_DA: dict,
                                deep_supervision_scales: Union[List, Tuple, None],
                                mirror_axes: Tuple[int, ...],
                                do_dummy_2d_data_aug: bool,
                                order_resampling_data: int = 1,
                                order_resampling_seg: int = 0,
                                border_val_seg: int = -1,
                                use_mask_for_norm: List[bool] = None,
                                is_cascaded: bool = False,
                                foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                ignore_label: int = None) -> AbstractTransform:
        return nnUNetTrainer.get_validation_transforms(deep_supervision_scales, is_cascaded, foreground_labels,
                                                       regions, ignore_label)

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        output = self.network(data)
        l = self.loss(output, target, self.current_epoch)

        # Skip batch if loss is NaN/Inf to prevent parameter corruption
        if not torch.isfinite(l):
            print(f"[WARNING] Non-finite loss ({l.item()}) at epoch {self.current_epoch}. Skipping batch.")
            return {'loss': np.nan}

        l.backward()
        # Tight grad clip for regression: prevents a single bad batch from corrupting parameters.
        # nnUNet default 12 is calibrated for segmentation; for unbounded z-score regression 6.0 is safer.
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 6.0)
        self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        output = self.network(data)
        del data
        l = self.loss(output, target, self.current_epoch)
        return {'loss': l.detach().cpu().numpy(), 'tp_hard': 0, 'fp_hard': 0, 'fn_hard': 0}

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        loss_here = np.mean(outputs_collated['loss'])
        self.logger.log('val_losses', loss_here, self.current_epoch)

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))

        epoch_duration = self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - \
                         self.logger.my_fantastic_logging['epoch_start_timestamps'][-1]
        self.print_to_log_file(f"Epoch time: {np.round(epoch_duration, decimals=2)} s")

        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        best_metric = 'val_losses'
        if self._best_ema is None or self.logger.my_fantastic_logging[best_metric][-1] < self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging[best_metric][-1]
            self.print_to_log_file(f"Yayy! New best EMA MAE: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1
