export HF_HOME="."
export HF_TOKEN=""
export CUDA_VISIBLE_DEVICES=0 

python main_pretrain.py --batch_size 1 \
    --accum_iter 2 \
    --mask_ratio 0.875 \
    --epochs 30 \
    --warmup_epochs 1 \
    --blr 4e-6 \
    --use_gigapath_model

