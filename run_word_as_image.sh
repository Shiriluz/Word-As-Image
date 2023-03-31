#!/bin/bash

set -e

USE_WANDB=0 # CHANGE IF YOU WANT WANDB
WANDB_USER="none"

EXPERIMENT=conformal_0.5_dist_pixel_100_kernel201

CONCEPT=BUNNY
WORD=BUNNY
fonts=(KaushanScript-Regular)
for j in "${fonts[@]}"
do
    letter_=("Y")
    SEED=0
    for i in "${letter_[@]}"
    do
        echo "$i"
        font_name=$j
        ARGS="--experiment $EXPERIMENT --optimized_letter ${i} --seed $SEED --font ${font_name} --use_wandb ${USE_WANDB} --wandb_user ${WANDB_USER}"
        CUDA_VISIBLE_DEVICES=0 python code/main.py $ARGS --semantic_concept "${CONCEPT}" --word "${WORD}"
    done
done

