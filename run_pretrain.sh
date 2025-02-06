accelerate launch \
        --config_file /your/path/configs/hf_accelerate_config.yaml \
        ./scripts/pretrain.py \
        --config_path /your/path/configs/pretrain_configs.yaml \
        --d_model 512 \
        --dim_feedforward 2048 \
        --num_heads 8 \
        --num_layers 6 \
        --codebook_dim 64 \
        --codebook_size 256 \
        --mask_ratio 0.3 \
        --num_quantizer 1 \
        --seed 0