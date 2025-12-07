#!/bin/bash

SUBDIR=cse599o_alignment
LOG_DIR=$SUBDIR/logs/async
mkdir -p $LOG_DIR

timestamp=$(date +"%Y%m%d_%H%M%S")

export PYTHONUNBUFFERED=1

ks=(1 2 4 8)
for k in "${ks[@]}"; do
    uv run $SUBDIR/train_grpo_ray_colocated.py --keywords_file $SUBDIR/prompts/keywords.txt \
        --train_val_kw_split_dir $SUBDIR/prompts/ \
        --ckpt_file $SUBDIR/ckpt/model.pt \
        --result_dir $SUBDIR/results/async/$timestamp/k$k \
        --steps 16 \
        --steps_per_rollout $k \
        --prompts_per_batch 8 \
        --ckpt_interval 100 \
        --verbose > $LOG_DIR/$timestamp-k$k.log 2>&1
        # --monitor_kl \
done
