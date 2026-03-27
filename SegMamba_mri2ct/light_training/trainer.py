import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from light_training.utils.lr_scheduler import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from monai.data import DataLoader
import argparse
from .launch import launch_dist
from monai.utils import set_determinism
from .sampler import SequentialDistributedSampler, distributed_concat

from torch.cuda.amp import GradScaler
from torch import nn
import time

def collate_with_none_filter(batch):
    """Custom collate function that filters out None values from nested dicts."""
    def filter_none(obj):
        if isinstance(obj, dict):
            return {k: filter_none(v) for k, v in obj.items() if v is not None}
        elif isinstance(obj, (list, tuple)):
            return type(obj)([filter_none(item) for item in obj if item is not None])
        return obj
    
    filtered_batch = [filter_none(item) for item in batch]
    # Fall back to default collate after filtering
    from torch.utils.data._utils.collate import default_collate
    try:
        return default_collate(filtered_batch)
    except Exception:
        # If still fails, return the filtered batch as-is
        return filtered_batch

class dummy_context(object):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class Trainer:
    def __init__(self, env_type,
                 max_epochs,
                 batch_size,
                 device="cpu",
                 val_every=1,
                 num_gpus=1,
                 logdir="./logs/",
                 master_ip='localhost',
                 master_port=17750,
                 training_script="train.py",
                 train_process=12,
                 ):
        assert env_type in ["pytorch", "ddp", "DDP"], f"not support this env_type: {env_type}"
        self.env_type = env_type
        self.val_every = val_every
        self.max_epochs = max_epochs
        self.ddp = False
        self.num_gpus = num_gpus
        self.device = device
        self.local_rank = 0
        self.batch_size = batch_size
        self.not_call_launch = True
        self.logdir = logdir
        self.scheduler = None 
        self.model = None
        self.auto_optim = True
        self.warmup = 0.0
        self.scheduler_type = None

        self.optimizer = None 
        self.patch_size = None 
        self.train_full_dataset_epoch = True
        self.num_step_per_epoch = 250 // self.num_gpus
        self.val_number = 100 // self.num_gpus
        self.augmentation = True
        self.num_patches_per_batch = 1
        self.train_process = train_process
        self.print_time = False

        # Disabled GradScaler to prevent NaN issues with mixed precision
        self.grad_scaler = None

        torch.backends.cudnn.enabled = True

        gpu_count = torch.cuda.device_count()
        if num_gpus > gpu_count:
            print(f"{num_gpus} GPUs were asked. {gpu_count} are detected in your hardware. Exiting...")
            os._exit(0)

        if env_type == "DDP" or env_type == "ddp":
            self.ddp = True
            self.get_dist_args()
            if not self.not_call_launch:
                launch_dist(env_type=env_type,
                            num_nodes=1,
                            gpus_per_node=num_gpus,
                            master_addr=master_ip,
                            master_port=master_port,
                            training_script=training_script,
                            )
                os._exit(1)
            self.initialize_distributed()

    def initialize_distributed(self):
        """Initialize torch.distributed."""
        if self.env_type == 'pytorch':
            self.print_rank_0('No need to initialize')
            return
        if self.env_type == 'DDP' or "deepspeed" in self.env_type:

            if self.local_rank is not None:
                device = self.local_rank
            torch.cuda.set_device(device)
            # Call the init process
            init_method = 'env://'
            torch.distributed.init_process_group(
                backend='nccl',
                init_method=init_method)
            self.world_size = torch.distributed.get_world_size()

            print(f"world size is {self.world_size}")

    def get_dataloader(self, dataset, shuffle=False, batch_size=1, train=True):
        if dataset is None :
            return None
        if self.env_type == 'pytorch':
            return DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=12)
        else :
            if not train:
                sampler = SequentialDistributedSampler(dataset, batch_size=batch_size)

            else :
                sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
            return DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=12, 
                                sampler=sampler, 
                                drop_last=True)

    def get_multi_processor_loader(self, train_ds, val_ds):
        from .augment.multi_processor import LimitedLenWrapper
        from .augment.train_augment import get_train_transforms, get_validation_transforms, get_train_transforms_noaug, get_train_transforms_nomirror, get_train_transforms_onlymirror, get_train_transforms_onlyspatial
        from light_training.dataloading.base_data_loader import DataLoaderMultiProcess

        assert self.patch_size != None 
        if self.augmentation:
            if self.augmentation == "nomirror":
                print(f"use augmentation: no mirror")
                tr_transforms = get_train_transforms_nomirror(patch_size=self.patch_size, mirror_axes=[0, 1, 2])
            elif self.augmentation == "onlymirror":
                print(f"use augmentation: only mirror")
                tr_transforms = get_train_transforms_onlymirror(patch_size=self.patch_size, mirror_axes=[0, 1, 2])
            elif self.augmentation == "onlyspatial":
                print(f"use augmentation: only spatial")
                tr_transforms = get_train_transforms_onlyspatial(patch_size=self.patch_size, mirror_axes=[0, 1, 2])
        
            else :
                tr_transforms = get_train_transforms(patch_size=self.patch_size, mirror_axes=[0, 1, 2])
        else:
            tr_transforms = get_train_transforms_noaug(patch_size=self.patch_size, mirror_axes=[0, 1, 2])

        val_transforms = get_validation_transforms()

        #train_loader = DataLoader(train_ds, num_workers=1, drop_last=True, shuffle=True, batch_size=self.batch_size)
        train_loader = DataLoaderMultiProcess(train_ds,
                                              batch_size=self.batch_size,
                                              num_patches_per_batch=self.num_patches_per_batch,
                                              patch_size=self.patch_size,
                                              print_time=self.print_time)
                                          
        if self.train_full_dataset_epoch:
            self.num_step_per_epoch = len(train_ds) // (self.batch_size*self.num_gpus)

        data_generator = LimitedLenWrapper(self.num_step_per_epoch, data_loader=train_loader, 
                                           transform=tr_transforms,
                                             num_processes=self.train_process, num_cached=6, seeds=None,
                                             pin_memory=True, wait_time=0.02)
        if val_ds is None:
            val_data_generator = None 
        else :
            val_loader = DataLoaderMultiProcess(val_ds, 
                                                batch_size=1,
                                                patch_size=self.patch_size,
                                                oversample_foreground_percent=1.0)
            
            val_data_generator = LimitedLenWrapper(self.val_number, data_loader=val_loader, transform=val_transforms,
                                                num_processes=6, num_cached=3, seeds=None,
                                                pin_memory=True, wait_time=0.02)
        return data_generator, val_data_generator


    def get_dist_args(self):
        parser = argparse.ArgumentParser()
        # parser.add_argument('--local_rank', type=int, default = 0, help="local_rank")
        parser.add_argument('--not_call_launch',
                            action='store_true',
                            help="not call launch!")
        ds_args = parser.parse_args()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        print(f"self.local_rank is {self.local_rank}")
        self.not_call_launch = ds_args.not_call_launch
        self.device = self.local_rank

    def to_device(self, batch):
        if isinstance(batch, dict):
            for k, v in batch.items():
                if isinstance(batch[k], np.ndarray):
                    batch[k] = torch.from_numpy(batch[k])

                if (isinstance(batch[k], torch.Tensor) or isinstance(batch[k], torch.FloatTensor)):
                    batch[k] = batch[k].to(self.device, non_blocking=True).contiguous()

        elif isinstance(batch, list) :
            batch = [torch.from_numpy(x) for x in batch if isinstance(x, np.ndarray)]
            batch = [x.to(self.device, non_blocking=True).contiguous() for x in batch if (isinstance(x, torch.Tensor) or isinstance(x, torch.FloatTensor))]

        elif isinstance(batch, np.ndarray):
            batch = torch.from_numpy(batch)
            batch = batch.to(self.device, non_blocking=True).contiguous()
        
        else :
            print("not support data type")
            exit(0)
        
        return batch 
    
    def validation_single_gpu(self, val_dataset, save_predictions=False):
        self.global_step = 0
        if self.ddp:
            print(f"single gpu model not support the ddp")
            exit(0)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True, collate_fn=collate_with_none_filter)
        if self.model is not None:
            if 'cuda' in self.device:
                self.model.cuda()
            else :
                self.model.to(self.device)
            self.model.eval()
        
        val_out_list = []
        
        for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            batch = self.before_data_to_device(batch)
            batch = self.to_device(batch)

            with torch.no_grad():
                val_out = self.validation_step(batch, save_predictions=save_predictions)
                assert val_out is not None 
                self.global_step += 1

            val_out_list.append(val_out)

        if len(val_out_list) == 0:
            mean_val_out = 0.0
        else:
            mean_val_out = float(np.mean(val_out_list))

        print(f"\n====== Mean Validation Output over {len(val_out_list)} cases: {mean_val_out:.4f} ======\n")
        

    def validate(self):
        val_outputs = []
        if self.global_step % self.val_every == 0 \
                and self.val_loader is not None :
            if self.model is not None:
                self.model.eval()
            if self.ddp:
                torch.distributed.barrier()
            outputs_split = None 
            # for idx, batch in tqdm(enumerate(self.val_loader), total=len(self.val_loader)):
            for i in tqdm(range(len(self.val_loader)), total=len(self.val_loader)):
                batch = next(self.val_loader)
                
                batch = self.before_data_to_device(batch)

                batch = self.to_device(batch)

                with torch.no_grad():
                    val_out = self.validation_step(batch)
                    assert val_out is not None 
                    if type(val_out) is not list and type(val_out) is not tuple:
                        val_out = [val_out]

                    if outputs_split is None:
                        outputs_split = [[] for i in range(len(val_out))]

                    for i, v in enumerate(val_out):
                        outputs_split[i].append(v)

                # val_outputs.append(val_out)

            ## 先汇总结果。
            if self.ddp:
                val_outputs = torch.tensor(val_outputs).cuda(self.local_rank)
                torch.distributed.barrier()    
                val_outputs_merge = []
                for i in range(len(outputs_split)):
                    val_outputs = torch.tensor(outputs_split[i]).cuda(self.local_rank)
                    val_outputs_merge.append(distributed_concat(val_outputs, num_total_examples=len(self.val_loader) * self.num_gpus))

                # val_outputs = distributed_concat(val_outputs, num_total_examples=len(self.val_loader.sampler.dataset))
                # val_outputs = distributed_concat(val_outputs, num_total_examples=len(self.val_loader) * self.num_gpus)
            else :
                val_outputs_merge = []
                for i in range(len(outputs_split)):
                    val_outputs = torch.tensor(outputs_split[i])
                    val_outputs_merge.append(val_outputs)
                # val_outputs = torch.tensor(val_outputs)

            if self.local_rank == 0:
                if len(val_outputs_merge) == 1:
                    val_outputs_merge = val_outputs_merge[0]
                self.validation_end(val_outputs_merge)
                # self.validation_end(val_outputs)
                
    def train(self,
                train_dataset,
                val_dataset=None,
              ):
        print(f"augmentation: {self.augmentation}")
        assert self.patch_size is not None, "please define the patch_size"

        set_determinism(42 + self.local_rank)
        if self.model is not None:
            print(f"check model parameter: {next(self.model.parameters()).sum()}, keep model parameters on different processes consistent")
            para = sum([np.prod(list(p.size())) for p in self.model.parameters()])
            if self.local_rank == 0:
                print(f"model parameters is {para / 1000 / 1000}M ")        
                
        self.global_step = 0
        if self.env_type == "pytorch":
            if 'cuda' in self.device:
                self.model.cuda()
            else :
                self.model.to(self.device)
            print("Model loaded to device:", self.device)

        elif self.ddp:
            if self.model is not None:
                self.model.cuda(self.local_rank)
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
                self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                                    device_ids=[self.local_rank],
                                                                    output_device=self.local_rank,
                                                                    find_unused_parameters=True)  
        else :
            print("not support env_type")
            exit(0)

        # self.train_loader = self.get_dataloader(train_dataset, shuffle=True, batch_size=self.batch_size)
        self.train_loader, self.val_loader = self.get_multi_processor_loader(train_dataset, val_dataset)
        
        self.max_steps = self.max_epochs * len(self.train_loader)

        print(f"Max step number is {self.max_steps}")

        if self.scheduler_type == "cosine_with_warmup":
            if self.warmup == 0.0:
                self.warmup = 0.1
            assert self.warmup < 1 and self.warmup > 0
            warmup_steps = self.max_steps * self.warmup
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=self.max_steps)
            print(f"warmup steps is {warmup_steps}")
        elif self.scheduler_type == "constant_with_warmup":
            if self.warmup == 0.0:
                self.warmup = 0.1
            assert self.warmup < 1 and self.warmup > 0
            warmup_steps = self.max_steps * self.warmup
            self.scheduler = get_constant_schedule_with_warmup(self.optimizer,
                                                num_warmup_steps=warmup_steps,
                                                )
            print(f"warmup steps is {warmup_steps}")

        elif self.scheduler_type == "poly_with_warmup":
            if self.warmup == 0.0:
                self.warmup = 0.1
            assert self.warmup < 1 and self.warmup > 0
            warmup_steps = self.max_steps * self.warmup
            self.scheduler = get_polynomial_decay_schedule_with_warmup(self.optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=self.max_steps
                                                )
            print(f"warmup steps is {warmup_steps}")
        
        elif self.scheduler_type == "poly":
            from light_training.utils.lr_scheduler import PolyLRScheduler
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            print(f"initial lr is {lr}")
            self.scheduler = PolyLRScheduler(self.optimizer, initial_lr=lr, max_steps=self.max_steps)
            print(f"scheduler_type is poly, warmup steps is {0}")
        if not hasattr(self, "resume_epoch") or self.resume_epoch is None:
            self.resume_epoch = 0

        # Advance scheduler to the correct LR position when resuming mid-training.
        # Without this the scheduler always starts fresh at initial_lr regardless of resume_epoch.
        if self.resume_epoch > 0 and self.scheduler is not None:
            resume_steps = self.resume_epoch * self.num_step_per_epoch
            if hasattr(self.scheduler, 'ctr'):
                # PolyLRScheduler: set counter directly and update optimizer LR in one shot
                self.scheduler.ctr = resume_steps
                new_lr = (self.scheduler.initial_lr
                          * (1 - resume_steps / self.scheduler.max_steps) ** self.scheduler.exponent)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"[RESUME] Poly scheduler advanced to step {resume_steps}, LR={new_lr:.6e}")
            else:
                # LambdaLR-based schedulers (cosine, constant, poly_with_warmup):
                # step() is a pure LR computation with no side-effects so fast-forward is cheap
                for _ in range(resume_steps):
                    self.scheduler.step()
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                print(f"[RESUME] Scheduler fast-forwarded {resume_steps} steps, LR={lr:.6e}")

            print(f"[INFO] Training will start from epoch {self.resume_epoch}")

        for epoch in range(self.resume_epoch, self.max_epochs):
            self.epoch = epoch 
            if self.ddp:
                torch.distributed.barrier()
            self.train_epoch(
                            epoch,
                            )
            if (self.epoch + 1) % self.val_every == 0:
                start_time = time.time() 
                self.validate()
                elapsed = time.time() - start_time
                hours = int(elapsed // 3600)
                minutes = int((elapsed % 3600) // 60)
                seconds = int(elapsed % 60)
                print(f"Validation time: {hours:02d}:{minutes:02d}:{seconds:02d}")
            if (self.epoch + 1) % 100 == 0:
                torch.save(self.model.state_dict(), os.path.join(self.model_save_path, f"checkpoint_ep{self.epoch}.pt"))
                print(f"Model saved on checkpoint_ep{self.epoch}.pt")
                
            if self.model is not None:
                self.model.train()
        print("Training process finished")    

    def before_data_to_device(self, batch_data):
        return batch_data 
    
    def train_epoch(self, 
                    epoch,
                    ):
        epoch_loss = 0.0
        start_time = time.time()
        if self.model is not None:
            self.model.train()
        # if self.local_rank == 0:
        with tqdm(total=self.num_step_per_epoch, disable=(self.local_rank != 0)) as t:
            for i in range(self.num_step_per_epoch):
                self.global_step += 1
                t.set_description('Epoch %i' % epoch)

                if self.print_time:
                    s = time.time()
                batch = next(self.train_loader)
                if self.print_time:
                    e = time.time()
                    print(f"get batch time is {e - s}")

                batch = self.before_data_to_device(batch)

                batch = self.to_device(batch)

                if self.model is not None:
                    for param in self.model.parameters(): param.grad = None
                
                if not self.auto_optim:
                    loss = self.training_step(batch)
                else:
                    if self.print_time:
                        s = time.time()
                    loss = self.training_step(batch)
                    if self.print_time:
                        e = time.time()
                        print(f"training step time is {e - s}")
                    
                    if self.print_time:
                        s = time.time()

                    # Check for NaN before backward pass
                    if torch.isnan(loss):
                        print(f"[WARNING] NaN loss detected at step {self.global_step}. Skipping this batch.")
                        continue
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
                    self.optimizer.step()

                    if self.print_time:
                        e = time.time()
                        print(f"backward time is {e - s}")
    
                    if self.scheduler is not None:
                        self.scheduler.step()
                    lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                    
                    t.set_postfix(loss=loss.item(), lr=lr)
                epoch_loss += loss.item()
                t.update(1)
        avg_loss = epoch_loss / self.num_step_per_epoch
        elapsed = time.time() - start_time

        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        if self.local_rank == 0:
            print(f"[Epoch {epoch}] Average training loss: {avg_loss:.6f}, Epoch Training time: {hours:02d}:{minutes:02d}:{seconds:02d}")
            
                
    def training_step(self, batch):
        raise NotImplementedError
    
    def validation_step(self, batch):
        raise NotImplementedError

    def validation_end(self, mean_val_outputs, val_outputs):
        pass 

    def log(self, k, v, step):
        if self.local_rank == 0:
            self.writer.add_scalar(k, scalar_value=v, global_step=step)

    def log_dict(self, dict_, step):
        if self.local_rank == 0:
            for k, v in dict_.items():
                self.writer.add_scalar(k, scalar_value=v, global_step=step)
                
    def load_state_dict(self, weight_path, strict=True):
        sd = torch.load(weight_path, map_location="cpu")
        if "module" in sd :
            sd = sd["module"]
        new_sd = {}
        for k, v in sd.items():
            k = str(k)
            new_k = k[7:] if k.startswith("module") else k 
            new_sd[new_k] = v 

        self.model.load_state_dict(new_sd, strict=strict)
        
        print(f"model parameters are loaded successed.")

