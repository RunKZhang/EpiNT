import torch
import os
import torch.nn as nn
import pandas as pd
import numpy as np
import time

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from .base import Base
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report
from ..utils.gradients import compute_gradient_norm

class Finetune(Base):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.method_name = 'Finetune'
        self.run_name = self.generate_run_name()
        self.results_dir = self._create_results_dir(self.method_name, self.run_name)

        # init dataloader, model, optimizer, lr_scheduler, criterion
        self.train_dataloader, self.val_dataloader, self.class_weights = self._get_downstream_dataloader(self.args.downstream_dataset, self.args.train_ratio,
                                                                                                         num_workers=self.args.num_workers)
        
        # make the results dict
        self.cls_metrics = self.get_classification_metrics()
        column_names = ['Epoch', 'Train Loss', 'Validation Loss']
        for key in self.cls_metrics.keys():
            column_names.append(key)
        self.metric_df = pd.DataFrame(columns = column_names)

        self.model = self._build_model()
        self.model = self.load_weights() # load weights

        self.optimizer = self._select_optimizer(self.model)
        self.lr_scheduler = self._init_lr_scheduler(self.args.lr_scheduler_type)
        
             
        # setup accelerator and wrap them
        proj_config = ProjectConfiguration(project_dir=self.results_dir, logging_dir='aim_url', automatic_checkpoint_naming=True, total_limit=1)
        self.accelerator = Accelerator(log_with='aim', 
                                       project_config=proj_config)
        self.accelerator.init_trackers(self.run_name, config=vars(self.args), init_kwargs={"aim": {"experiment": self.args.experiment_name}})
        # self.device = self.accelerator.device
        self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler
            )
        self.criterion = self._select_criterion(loss_type="ce", reduction="mean")
        self.print_model_summary()

        # if self.accelerator.is_main_process:
        self.accelerator.print('\n')
        self.accelerator.print(f'running with config: {self.args}')
        self.accelerator.print('\n')
        self.accelerator.print(f'run_name: {self.run_name}')
        self.accelerator.print('\n')
        self.accelerator.print(f'Estimated train loader and vali loader length: {len(self.train_dataloader)} {len(self.val_dataloader)}')
        self.accelerator.print('\n')
        self.accelerator.print(f"Accelerator state from the current environment:\n{self.accelerator.state}")
        self.accelerator.print('\n')
        self.accelerator.print(f'Class weights: {self.class_weights}')
        self.accelerator.print('\n')

    def train(self):
        cur_epoch = 0
        best_eval_loss = np.inf
        
        while cur_epoch < self.args.max_epoch:
            self.accelerator.print(f">>>>>>>>>>>  {cur_epoch} / {self.args.max_epoch} >>>>>>>>>>>>>>")

            self.model.train()
            train_loss = []
            eval_loss = []

            # train the model
            for _, (batch_x, y) in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                batch_x = batch_x.to(self.accelerator.device)
                y = y.to(self.accelerator.device)
                pred = self.model(batch_x)
                loss = self.criterion(pred, y)

                # use accelerate
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.lr_scheduler.step() # use onecycle lr scheduler

                train_loss.append(loss.item())

                if self.args.debug:
                    gradient_norm = compute_gradient_norm(self.model)
            
            train_loss = np.array(train_loss).mean()

            # validate the model
            eval_loss, metric_dict = self.validation()

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
            self.accelerator.save_state() # save after each epoch, compare the last epoch's performance 

            # Track the loss at each epoch
            self.accelerator.log({"train_loss": train_loss,
                                  "eval_loss": eval_loss, 
                                 }, log_kwargs={"aim": {"epoch": cur_epoch}})
            
            # print the metrics
            for key, value in metric_dict.items():
                self.accelerator.print(f"{key}: {value}")
            
            # combine metrics and loss to a dataframe
            resulist = [cur_epoch, train_loss, eval_loss]
            for key, value in metric_dict.items():
                resulist.append(value)
                
            self.metric_df.loc[len(self.metric_df)] = resulist

            # get confusion mat
            conf_mat = metric_dict['Confusion Matrix']
            del metric_dict['Confusion Matrix']

            # modify accuracy per class
            # if self.args.seizure_task == 'seiz_pred':
            #     metric_dict['Accuracy per class'] = np.array(metric_dict['Accuracy per class']).mean()
            self.accelerator.log(metric_dict, log_kwargs={"aim": {"epoch": cur_epoch}})
            self.accelerator.log({"lr": self.optimizer.param_groups[0]["lr"]}, log_kwargs={"aim": {"epoch": cur_epoch}})

            cur_epoch += 1
        
        if self.accelerator.is_main_process:
            self.metric_df.to_csv(os.path.join(self.results_dir, 'metric.csv'), index=False)
            conf_mat = pd.DataFrame(conf_mat)
            conf_mat.to_csv(os.path.join(self.results_dir, 'confusion_matrix.csv'), index=False)
            aim_tracker = self.accelerator.get_tracker(name='aim', unwrap=True)
            
            # add tag
            tag_string = f'{self.args.experiment_name}' # not tested
            aim_tracker.add_tag(tag_string)

        self.accelerator.end_training()

    def validation(self):
        eval_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for _, (batch_x, y) in enumerate(self.val_dataloader):
                batch_x = batch_x.to(self.accelerator.device)
                y = y.to(self.accelerator.device)
                pred = self.model(batch_x)
                pred, y = self.accelerator.gather_for_metrics((pred, y)) # gather the data from all processes, otherwise cause error
                loss = self.criterion(pred, y)
                
                eval_loss.append(loss.item())
                preds.append(pred)
                trues.append(y)
        
        eval_loss = np.array(eval_loss)
        average_loss = np.average(eval_loss)

        # calculate classification metrics
        metric_dict = {}

        preds = torch.cat(preds, dim=0)[:,-1] # get Positive class
        # preds = torch.cat(preds, dim=0) # used when bce loss
        trues = torch.cat(trues, dim=0)
        # trues = trues.to(torch.long) # used when bce loss
        
        self.accelerator.print(f'preds value range: {preds.min().item()} {preds.max().item()}')
        self.accelerator.print(f'trues value range: {trues.min().item()} {trues.max().item()}')

        for key, value in self.cls_metrics.items():
            tmp_fn = value.to(self.accelerator.device)
            metric_value = tmp_fn(preds, trues)
            metric_dict[key] = metric_value.detach().cpu().numpy()

        self.model.train()
        return average_loss, metric_dict