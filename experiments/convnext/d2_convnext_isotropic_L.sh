python convnext/run_with_submitit.py --nodes 8 --ngpus 4 \
    --model d2_convnext_isotropic_large --drop_path 0.5 \
    --batch_size 64 --lr 4e-3 --update_freq 2 \
    --warmup_epochs 50 --model_ema true --model_ema_eval true \
    --num_workers 15 \
    --use_amp True \
    --compile \
    --high-precision-matmul