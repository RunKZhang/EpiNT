import os
import sys
import argparse
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from EpiNT.utils.config import Config
from EpiNT.utils.tools import parse_config, seed_everything
from EpiNT.tasks.finetune import Finetune

def finetune(args):   
    # Control randomness
    seed_everything(args.seed)
    
    task_obj = Finetune(args=args)
    task_obj.train()

if __name__ == "__main__":
    default_config_path = "/your/path/configs/default.yaml"

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="Path to config file")

    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--dim_feedforward", type=int, default=1024, help="Feedforward dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--codebook_size", type=int, default=512, help="Codebook size")
    parser.add_argument("--codebook_dim", type=int, default=256, help="Codebook dimension")

    parser.add_argument("--seed", type=int, default=0, help="Seed for randomness")
    parser.add_argument("--run_name", type=str, help="Saved run name")

    # parser.add_argument("--ablation", type=str, default='None', help="Abalation study")
    
    # parser.add_argument("--experiment_name", type=str, default='last_layer', help="Experiment name") # linear_probing or finetuning or last_layer
    # parser.add_argument("--seizure_task", type=str, default='soz_loc', help="Task name") #'soz_loc' 'seiz_pred' 'seiz_detec' 'ied_detec' 'hfo_ied_detec1' 'hfo_ied_detec2'

    args = parser.parse_args()
    config = Config(args, config_file_path=args.config_path, default_config_file_path=default_config_path, verbose=False).parse()
    args = parse_config(config)
    
    finetune(args)
