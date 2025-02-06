import torch
import random
import numpy as np

from argparse import Namespace

class NamespaceWithDefaults(Namespace):
    @classmethod
    def from_namespace(cls, namespace):
        new_instance = cls()
        for attr in dir(namespace):
            if not attr.startswith("__"):
                setattr(new_instance, attr, getattr(namespace, attr))
        return new_instance

    def getattr(self, key, default=None):
        return getattr(self, key, default)

def parse_config(config: dict) -> NamespaceWithDefaults:
    args = NamespaceWithDefaults(**config)
    return args

def seed_everything(seed):
    random.seed(seed) # Python random module
    np.random.seed(seed) # numpy
    torch.manual_seed(seed) # pytorch
    torch.cuda.manual_seed(seed) # for using all GPUs
    torch.cuda.manual_seed_all(seed) # if using multi-GPU
    torch.backends.cudnn.deterministic = True # fix some algorithms
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False