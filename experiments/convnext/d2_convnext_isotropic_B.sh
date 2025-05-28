python convnext/run_with_submitit.py --nodes 4 --ngpus 4 \
    --model d2_convnext_isotropic_base --drop_path 0.2 \
    --batch_size 256 --lr 4e-3 --update_freq 1 \
    --warmup_epochs 50 --model_ema true --model_ema_eval true \
    --layer_scale_init_value 0 \
    --num_workers 15 \
    --tgpu A100 \
    --use_amp True \
    --compile \
    --high-precision-matmul