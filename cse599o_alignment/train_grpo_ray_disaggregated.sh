#!/bin/bash

timestamp=$(date +"%Y%m%d_%H%M%S")

SUBDIR=cse599o_alignment
LOG_DIR=$SUBDIR/logs/disaggregated
RESULT_DIR=$SUBDIR/results/disaggregated/$timestamp
mkdir -p $LOG_DIR
mkdir -p $RESULT_DIR


export PYTHONUNBUFFERED=1

steps=64
prompts_per_batch=1
split_dir=$SUBDIR/prompts/
keywords_file=$SUBDIR/prompts/keywords.txt
ckpt_file=$SUBDIR/ckpt/model.pt

uv run $SUBDIR/train_grpo_ray_disaggregated.py --keywords_file $keywords_file \
    --train_val_kw_split_dir $split_dir \
    --ckpt_file $ckpt_file \
    --result_dir $RESULT_DIR/disaggregated \
    --steps $steps \
    --prompts_per_batch $prompts_per_batch \
    --profile \
    --verbose > $LOG_DIR/$timestamp.log 2>&1

ks=(1 4)
for k in "${ks[@]}"; do
    coloc_result_dir=$RESULT_DIR/colocated_k$k
    uv run $SUBDIR/train_grpo_ray_colocated.py --keywords_file $keywords_file \
        --train_val_kw_split_dir $split_dir \
        --ckpt_file $ckpt_file \
        --result_dir $coloc_result_dir \
        --steps $steps \
        --steps_per_rollout $k \
        --prompts_per_batch $prompts_per_batch \
        --ckpt_interval 100 \
        --profile \
        --verbose > $LOG_DIR/$timestamp-k$k.log 2>&1
done
