accelerate launch \
        --config_file /your/path/configs/hf_accelerate_config.yaml \
        ./scripts/finetune.py \
        --config_path /your/path/configs/finetune_configs.yaml \
        --run_name 20241226_015111 \
        --d_model 512 \
        --dim_feedforward 2048 \
        --num_heads 8 \
        --num_layers 6 \
        --codebook_dim 64 \
        --codebook_size 512 \
        --seed 7 \