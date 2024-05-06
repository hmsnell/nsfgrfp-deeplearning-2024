#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: carp_davinci003.sh


PROJECT_PATH=/Users/yzhu194/Documents/Brown/CSCI2470/FinalProject/nsfgrfp-deeplearning-2024/code/CARP/
export PYTHONPATH="$PYTHONPATH:$PROJECT_PATH"


DATASET=csci1470
MODEL=gpt3_zeroshot
SETTING=carp


for seed in 2250
do
  echo "=============================================================================="
  echo "SEED IS " ${seed}
  echo ${DATASET} "-" ${MODEL} "-" ${SETTING}
  echo "=============================================================================="
  python3 ${PROJECT_PATH}/task/gpt3_text_cls.py \
  --seed ${seed} \
  --config_path ${PROJECT_PATH}/configs/${DATASET}/${MODEL}/${SETTING}.json \
  --step_idx 1-2-3-4
done