#!/bin/bash

timestamp=$(date +"%Y%m%d_%H%M%S")

SUBDIR=cse599o_alignment
LOG_DIR=$SUBDIR/logs/rdt
RESULT_DIR=$SUBDIR/results/rdt/$timestamp
mkdir -p $LOG_DIR
mkdir -p $RESULT_DIR


export PYTHONUNBUFFERED=1

steps=64
prompts_per_batch=1
split_dir=$SUBDIR/prompts/
keywords_file=$SUBDIR/prompts/keywords.txt
ckpt_file=$SUBDIR/ckpt/model.pt

# uv run $SUBDIR/train_grpo_ray_disaggregated.py --keywords_file $keywords_file \
#     --train_val_kw_split_dir $split_dir \
#     --ckpt_file $ckpt_file \
#     --result_dir $RESULT_DIR/disaggregated \
#     --steps $steps \
#     --prompts_per_batch $prompts_per_batch \
#     --profile \
#     --verbose > $LOG_DIR/$timestamp.log 2>&1

# uv run $SUBDIR/train_grpo_ray_disaggregated.py --keywords_file $keywords_file \
#     --train_val_kw_split_dir $split_dir \
#     --ckpt_file $ckpt_file \
#     --result_dir $RESULT_DIR/rdt \
#     --steps $steps \
#     --prompts_per_batch $prompts_per_batch \
#     --rdt \
#     --profile \
#     --verbose > $LOG_DIR/$timestamp-rdt.log 2>&1

transfer_dir=$RESULT_DIR/transfer_only
mkdir -p $transfer_dir
uv run $SUBDIR/benchmark_rdt.py --ckpt_file $ckpt_file \
    --result_dir $transfer_dir \
    --verbose > $LOG_DIR/$timestamp-transfer.log 2>&1