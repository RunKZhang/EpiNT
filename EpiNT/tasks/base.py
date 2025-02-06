import torch
import os
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import aim
import time
import random
import string
from torchinfo import summary
from safetensors.torch import load_file
from torchmetrics import Accuracy, Recall, CohenKappa, F1Score, AUROC, Precision, ConfusionMatrix, AveragePrecision
from accelerate.tracking import on_main_process
from EpiNT.data.make_datasets import create_dataloaders, create_downstream_dataloaders
from EpiNT.models.EpiNT import EpiNT
from EpiNT.models.EpiNT_ablation import EpiNT_ablation
from EpiNT.utils.optmis import LinearWarmupCosineLRScheduler
from EpiNT.data.constants import DOWNSTREAM_NUM_CLASS

class Base:
    def __init__(self, args):
        self.args = args
    
    def _build_model(self):
        if self.args.model_name == 'EpiNT':
            model = EpiNT(self.args)
        elif self.args.model_name == 'EpiNT_ablation':
            model = EpiNT_ablation(self.args)

        return model

    def _select_optimizer(self, model):
        if self.args.optimizer_name == "AdamW":
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.args.init_lr,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer_name == "Adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.args.init_lr,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer_name == "SGD":
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.args.init_lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )
        else:
            raise NotImplementedError(
                f"Optimizer {self.args.optimizer_name} not implemented"
            )
        return optimizer
    
    def _init_lr_scheduler(self, type: str = "linearwarmupcosinelr"):
        if type == "linearwarmupcosinelr":
            decay_rate = self.args.lr_decay_rate
            warmup_start_lr = self.args.warmup_lr
            warmup_steps = self.args.warmup_steps
            lr_scheduler = LinearWarmupCosineLRScheduler(
                optimizer=self.optimizer,
                max_epoch=self.args.max_epoch,
                min_lr=self.args.min_lr,
                init_lr=self.args.init_lr,
                decay_rate=decay_rate,
                warmup_start_lr=warmup_start_lr,
                warmup_steps=warmup_steps,
            )
        elif type == "onecyclelr":
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer,
                max_lr=self.args.init_lr,
                epochs=self.args.max_epoch,
                steps_per_epoch=len(self.train_dataloader),
                pct_start=self.args.pct_start,
                three_phase=self.args.three_phase,
            )
        elif type == "none":
            lr_scheduler = None
        
        return lr_scheduler

    def _get_dataloader(self, file_path, train_ratio=0.8, num_workers=4):
        batch_size = self.args.train_batch_size
        
        train_dataloaders, val_dataloaders = create_dataloaders(file_path, batch_size, train_ratio, num_workers=num_workers
                                                                    # mode=mode, train_ratio=train_ratio
                                                                )
        return train_dataloaders, val_dataloaders

    def _get_downstream_dataloader(self, file_path, train_ratio=0.8, num_workers=4):
        batch_size = self.args.train_batch_size
        seizure_task = self.args.seizure_task
        train_dataloader, val_dataloader, class_weights = create_downstream_dataloaders(file_path, batch_size, seizure_task=seizure_task, 
                                                                         train_ratio=train_ratio, num_workers=num_workers)

        return train_dataloader, val_dataloader, class_weights
    
    def _select_criterion(
        self, loss_type: str = "mse", reduction: str = "none", **kwargs
    ):
        if loss_type == "ce":
            weights = self.class_weights.to(self.accelerator.device)
            criterion = nn.CrossEntropyLoss(weight = weights, reduction=reduction)
        elif loss_type == "mse":
            criterion = nn.MSELoss(reduction=reduction)
        elif loss_type == 'nll':
            criterion = nn.NLLLoss(reduction=reduction)
        elif loss_type == "bce":
            criterion = nn.BCELoss(reduction=reduction)
        elif loss_type == "bce_with_logits":
            criterion = nn.BCEWithLogitsLoss(reduction=reduction)
        return criterion

    def _create_results_dir(self, method_name: str, run_name: str):
        results_path = os.path.join(
            self.args.RESULTS_DIR,
            method_name,
            self.args.model_name,
            run_name
            )
        if not os.path.exists(results_path):
            os.makedirs(results_path, exist_ok=True)
        return results_path

    def generate_run_name(self):
        current_time = time.localtime()
        formatted_time = time.strftime("%Y%m%d_%H%M%S", current_time)   
        final_string = f"{formatted_time}"
        
        return final_string

    def get_classification_metrics(self):
        recall = Recall(task='binary')
        auroc = AUROC(task='binary')
        auprc = AveragePrecision(task='binary')
        accuracy = Accuracy(task='binary')
        F1_score = F1Score(task='binary')
        kappa = CohenKappa(task='binary')
        precision = Precision(task='binary')
        conf_mat = ConfusionMatrix(task='binary')

        cls_metrics = {
            'Recall': recall,
            'AUROC': auroc,
            'Accuracy': accuracy,
            'AUPRC': auprc,
            'F1 Score': F1_score,
            'Cohen Kappa': kappa,
            'Precision': precision,
            'Confusion Matrix': conf_mat
        }

        return cls_metrics

    def mode_load(self, mode):
        if mode == 'linear_probing':
            # freeze the model, except head
            for name, param in self.model.named_parameters():
                if not name.startswith('head'):
                    param.requires_grad = False
        
            print('Model weights in linear probing mode')
        
        elif mode == 'finetuning':
            for name, param in self.model.named_parameters():
                param.requires_grad = True
            
            print('Model weights in full parameter fine tuning mode')
        
        elif mode == "last_layer":
            # freeze all weights except the last layer
            last_layer = self.args.num_layers - 1
            print(f'Last layer: {last_layer}')
            print(f'num_layers: {self.args.num_layers}')
            print(f'backbone_prefix: {backbone_prefix[self.args.model_name]}')
            for name, param in self.model.named_parameters():
                if not name.startswith('head'):
                    param.requires_grad = False
                if name.startswith(f'{backbone_prefix[self.args.model_name]}.{last_layer}'):
                    # print(f'name: {name}')
                    param.requires_grad = True
            
            print('Model weights in last layer fine tuning mode')

    def load_weights(self):
        if self.args.run_name == None:
            print('No model weights loaded, training from scratch')
        else:
            input_dir = os.path.join(self.args.RESULTS_DIR, 'Pretrain', self.args.model_name, self.args.run_name, 'checkpoints')
            ckp_name = os.listdir(input_dir)[0]
            input_dir = os.path.join(self.args.RESULTS_DIR, 'Pretrain', self.args.model_name, self.args.run_name, 'checkpoints', ckp_name)
            print(f'Loading model weights from {input_dir}')
            safetensor = load_file(os.path.join(input_dir, 'model.safetensors'))
            tobeload_keys = to_be_load_dict[self.args.model_name]
            filtered_state_dict = {k: v for k, v in safetensor.items() if k.split('.')[0] in tobeload_keys} # load only some weights
            self.model.load_state_dict(filtered_state_dict, strict=False)
            self.mode_load(self.args.experiment_name)
            # for name, param in self.model.named_parameters():
            #     print(name, param.requires_grad)
            
        return self.model
    
    def print_model_summary(self):
        if self.accelerator.is_main_process:
            batch_size = self.args.train_batch_size
            self.model.eval()
            summary(self.model, input_size=(batch_size, self.args.sequence_len))

    def debug_model_outputs(self, loss, outputs, batch_x):
        if (torch.any(torch.isnan(loss)) or 
            torch.any(torch.isinf(loss)) 
            # (loss < 1e-3 )
        ):
            print(f'Loss has NaN or Inf values: {loss}')
        
        if (torch.any(torch.isnan(outputs)) or
            torch.any(torch.isinf(outputs))
        ):
            print(f'Outputs has NaN or Inf values')
        
        illegal_encoder_grads = (
            torch.stack(
                [torch.isnan(p).any() or torch.isinf(p).any() for p in self.model.transformer_encoder.parameters()]
            )
            .any()
            .item()
        )
        illegal_head_grads = (
            torch.stack([torch.isnan(p).any() or torch.isinf(p).any() for p in self.model.head.parameters()])
            .any()
            .item()
        )
        illegal_patch_embedding_grads = (
            torch.stack(
                [
                    torch.isnan(p).any() or torch.isinf(p).any()
                    for p in self.model.input_embedding.parameters()
                ]
            )
            .any()
            .item()
        )

        illegal_grads = (
            illegal_encoder_grads or illegal_head_grads or illegal_patch_embedding_grads
        )

        if illegal_grads:
            print("Model has illegal gradients")
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                layer_norm = param.grad.data.norm(2)
                # print(f"{name}: {layer_norm}")

        return


to_be_load_dict = {
    'EEPT': ['embed', 'transformer_encoder', 'norm'],
    'EEPT_ablation': ['embed', 'transformer_encoder', 'norm'],
}

backbone_prefix = {
    'EEPT': 'transformer_encoder',
    'EEPT_ablation': 'transformer_encoder',
}