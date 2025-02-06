import argparse
from yaml import CLoader as Loader
from yaml import dump, load


class Config:
    def __init__(
        self,
        args,
        config_file_path="configs/config.yaml",
        default_config_file_path="configs/default.yaml",
        verbose: bool = True,
    ):
        """
        Class to read and parse the config.yml file
        """
        self.args = args
        self.config_file_path = config_file_path
        self.default_config_file_path = default_config_file_path
        self.verbose = verbose

        self.args_dict = vars(args)

    def parse(self):
        with open(self.config_file_path, "rb") as f:
            self.config = load(f, Loader=Loader)

        with open(self.default_config_file_path, "rb") as f:
            default_config = load(f, Loader=Loader)

        # Update the config with the args
        final_config = {**self.config, **self.args_dict}
        for key in default_config.keys():
            if final_config.get(key) is None:
                final_config[key] = default_config[key]
                if self.verbose:
                    print(f"Using default config for {key} : {default_config[key]}")

        return final_config

    def save_config(self):
        with open(self.config_file_path, "w") as f:
            dump(self.config, f)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('YES', 'True', 't', 'y'):
        return True
    elif v.lower() in ('NO', 'False', 'f', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')