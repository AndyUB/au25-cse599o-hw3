#!/bin/bash

SUBDIR=cse599o_alignment
LOG_DIR=$SUBDIR/logs/colocated
mkdir -p $LOG_DIR

timestamp=$(date +"%Y%m%d_%H%M%S")

uv run $SUBDIR/train_grpo_ray_colocated.py --keywords_file $SUBDIR/prompts/keywords.txt \
    --ckpt_file $SUBDIR/ckpt/model.pt \
    --verbose > $LOG_DIR/$timestamp.log 2>&1
