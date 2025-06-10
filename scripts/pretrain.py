import argparse
import torch


from epint.utils.config import Config
from epint.utils.tools import parse_config, seed_everything
from epint.tasks.pretrain import Pretrain

from accelerate.commands.launch import launch_command
def pretrain(args):   
    # Control randomness
    seed_everything(args.seed)
    
    task_obj = Pretrain(args=args)
    task_obj.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--default_config_path", type=str, help="Path to default config file")
    parser.add_argument("--pretrain_config_path", type=str, help="Path to pretrain config file")
    args = parser.parse_args()

    config = Config(args, config_file_path=args.pretrain_config_path, default_config_file_path=args.default_config_path, verbose=False).parse()
    args = parse_config(config)

    print(args)
    
    pretrain(args)
