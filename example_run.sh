#!/bin/bash

PYTHONPATH=python3
BASEPATH="./"
DATASET=agnews #agnews(0) dbpedia(0) imdb(3) amazon(3) yahoo(2) sst2(0) mnli-m(0) mnli-mm(0) cola(0)
TEMPLATEID=0 # 0 1 2 3
SEED=144 # 145 146 147 148
SHOT=0 # 0 1 10 20
VERBALIZER=ept #
CALIBRATION=""
VERBOSE=1
MODEL="roberta" # "roberta"
MODEL_NAME_OR_PATH="roberta-large" # "roberta-large" # "bert-base-uncased"
# RESULTPATH="results_agnews.txt"
OPENPROMPTPATH="."

cd $BASEPATH


rm -f ${DATASET}_${MODEL}_cos.pt

mkdir -p "results/$MODEL_NAME_OR_PATH"

i=12

CUDA_VISIBLE_DEVICES=4 $PYTHONPATH emb_prompt.py \
        --model $MODEL \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --result_file "results/$MODEL_NAME_OR_PATH/results_$DATASET.txt" \
        --openprompt_path $OPENPROMPTPATH \
        --dataset $DATASET \
        --template_id $TEMPLATEID \
        --seed $SEED \
        --verbose $VERBOSE \
        --verbalizer $VERBALIZER $CALIBRATION \
        --select $i
