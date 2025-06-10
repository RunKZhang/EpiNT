accelerate launch \
        --config_file ../configs/hf_accelerate_config.yaml \
        ./pretrain.py \
        --pretrain_config_path ../configs/pretrain_configs.yaml \
        --default_config_path ../configs/default.yaml \
