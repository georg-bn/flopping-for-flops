export HDF5_USE_FILE_LOCKING='FALSE'
export OMP_NUM_THREADS=1
source ~/.bashrc
conda activate deit2
cd /mimer/NOBACKUP/groups/snic2022-6-266/bokman/deit/
model=hybrid_deit_huge_patch14_LS
job_dir="out_dir/$model"
mkdir -p $job_dir

data_path="/mimer/NOBACKUP/groups/snic2022-6-266/data/imagenet"

python deit/run_with_submitit.py --nodes 8 --ngpus 4 \
    --model $model \
    --job_dir $job_dir \
    --batch-size 64 \
    --lr 3e-3 \
    --drop-path 0.5 \
    --epochs 400 \
    --weight-decay 0.02 \
    --sched cosine \
    --input-size 224 \
    --reprob 0.0 \
    --color-jitter 0.3 \
    --eval-crop-ratio 1.0 \
    --smoothing 0.0 \
    --warmup-epochs 5 \
    --drop 0.0 \
    --seed 0 \
    --opt fusedlamb \
    --warmup-lr 1e-6 \
    --mixup .8 \
    --cutmix 1.0 \
    --unscale-lr \
    --repeated-aug \
    --bce-loss \
    --ThreeAugment \
    --high-precision-matmul \
    --fused-attn \
    --use-amp \
    --no-zip-dataloader \
    --tgpu A100 \
    --compile \
    --num_workers 15 \

exit 0
