import torch
import os
import torch.nn as nn
import pandas as pd
import numpy as np
import time

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from .base import Base

from ..utils.gradients import compute_gradient_norm

def check_grad(module, grad_input, grad_output):
    for i, grad in enumerate(grad_output):
        if grad is not None:
            if torch.isnan(grad).any():
                print(f"NaN detected in gradient output of {module} at output {i}")
            if torch.isinf(grad).any():
                print(f"Inf detected in gradient output of {module} at output {i}")

class Pretrain(Base):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.method_name = 'Pretrain'
        self.run_name = self.generate_run_name()
        self.results_dir = self._create_results_dir(self.method_name, self.run_name)
        self.metric_df = pd.DataFrame(columns = ['Step', 'Train Loss', 'Validation Loss'])

        # init dataloader, model, optimizer, lr_scheduler, criterion
        self.train_dataloader, self.val_dataloader = self._get_dataloader(self.args.pretrain_dataset, self.args.train_ratio,
                                                                num_workers=self.args.num_workers)
        self.model = self._build_model()

        self.optimizer = self._select_optimizer(self.model)
        self.lr_scheduler = self._init_lr_scheduler(self.args.lr_scheduler_type)
             
        # setup accelerator and wrap them
        proj_config = ProjectConfiguration(project_dir=self.results_dir, 
                                           automatic_checkpoint_naming=True, total_limit=1)
        self.accelerator = Accelerator(project_config=proj_config)
        self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler
            )
        self.print_model_summary()

        self.accelerator.print('\n')
        self.accelerator.print(f'running with config: {self.args}')
        self.accelerator.print('\n')
        self.accelerator.print(f'run_name: {self.run_name}')
        self.accelerator.print('\n')
        self.accelerator.print(f'Estimated train loader and vali loader length: {len(self.train_dataloader)} {len(self.val_dataloader)}')
        self.accelerator.print('\n')
        self.accelerator.print(f"Accelerator state from the current environment:\n{self.accelerator.state}")
        self.accelerator.print('\n')

    def train(self):
        opt_steps = 0
        cur_epoch = 0
        best_eval_loss = np.inf
        eval_loss = 0.0
        
        while opt_steps < self.args.max_opt_steps and cur_epoch < self.args.max_epoch:
            self.model.train()
            interval_time = time.time()
            train_loss = []
            
            # print('hello world')
            for _, batch_x in enumerate(self.train_dataloader):
                # print('hello world')
                self.optimizer.zero_grad()
                batch_x = batch_x.to(self.accelerator.device)
                rec, loss = self.model(batch_x)
                    
                # use accelerate
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
                self.optimizer.step()
                self.lr_scheduler.step(cur_epoch=cur_epoch, cur_step=opt_steps)

                train_loss.append(loss.item())

                if self.args.debug:
                    gradient_norm = compute_gradient_norm(self.model)

                # if reach eval step, evaluate the model
                if opt_steps % self.args.interval == 0 and opt_steps != 0:
                    self.accelerator.print(f">>>>>>>>>>> {opt_steps} / {self.args.max_opt_steps} | {cur_epoch} / {self.args.max_epoch} >>>>>>>>>>>>>>")
                    train_loss = np.average(np.array(train_loss))
                    self.accelerator.print(f'Interval time: {time.time() - interval_time} seconds, train_loss: {train_loss}')
                    train_loss = []
                    
                    eval_start_time = time.time()
                    eval_loss = self.validation()
                    self.accelerator.print(f'Evaluation time: {time.time() - eval_start_time} seconds, eval_loss: {eval_loss}')

                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        self.accelerator.save_state()

                    interval_time = time.time() # reset interval time
                
                    # self.accelerator.print(f'Step: {opt_steps} | Epoch: {cur_epoch} | train_Loss: {loss.item()} | val_loss: {eval_loss}')

                self.metric_df.loc[len(self.metric_df)] = [opt_steps, loss.item(), eval_loss]
                
                opt_steps += 1
                
                if opt_steps >= self.args.max_opt_steps or cur_epoch >= self.args.max_epoch:
                    break

            cur_epoch += 1
        
        if self.accelerator.is_main_process:
            self.metric_df.to_csv(os.path.join(self.results_dir, 'metric.csv'), index=False)

            aim_tracker = self.accelerator.get_tracker(name='aim', unwrap=True)
            
            # add tag
            tag_string = f'{self.args.experiment_name}_{self.args.model_name}' # not tested
            aim_tracker.add_tag(tag_string)

        self.accelerator.end_training()

    def validation(self):
        eval_loss = []
        self.model.eval()
        with torch.no_grad():
            for _, batch_x in enumerate(self.val_dataloader):
                batch_x = batch_x.to(self.accelerator.device)
                rec, loss = self.model(batch_x)
                loss_value = self.accelerator.reduce(loss, reduction='mean')
                eval_loss.append(loss_value.item())
                
        eval_loss = np.array(eval_loss)
        average_loss = np.average(eval_loss)

        self.model.train()
        return average_loss


    
