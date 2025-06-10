accelerate launch \
        --config_file ../configs/hf_accelerate_config.yaml \
        ./finetune.py \
        --finetune_config_path ../configs/finetune_configs.yaml \
        --default_config_path ../configs/default.yaml \