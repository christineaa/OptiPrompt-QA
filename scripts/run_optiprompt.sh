#!/bin/bash

OUTPUTS_DIR=optiprompt-outputs
MODEL=fnlp/bart-base-chinese
RAND=none
REL=P1


DIR=${OUTPUTS_DIR}/${REL}
mkdir -p ${DIR}

python code/run_optiprompt.py \
    --relation_profile relation_metainfo/QA_relations.jsonl \
    --relation ${REL} \
    --model_name ${MODEL} \
    --do_train \
    --train_data cmrc2018_public/train.json \
    --dev_data cmrc2018_public/dev.json \
    --do_eval \
    --test_data cmrc2018_public/test.json \
    --output_dir ${DIR} \
    --random_init ${RAND} \
    --init_manual_template \
    --eval_per_epoch 1 \
    --num_epoch 20 \
    --learning_rate 5e-3 \
    --freeze
#    --debug

#python code/accumulate_results.py ${OUTPUTS_DIR}