#!/bin/bash

#conda create --name indolem
#conda activate indolem
#conda install -c pytorch pytorch
pip install seqeval
pip install transformers==2.9.0

export MAX_LENGTH=128
export BERT_MODEL="indolem/indobert-base-uncased"

for f in 1 2 3 4 5
do

export FOLD=0$f
export OUTPUT_DIR=indoner$FOLD
export BATCH_SIZE=32
export NUM_EPOCHS=30
export SAVE_STEPS=750
export SEED=1
export DATA_DIR=./data/nerugm


cat $DATA_DIR/train.$FOLD.tsv | tr '\t' ' '  | tr '  ' ' ' > train.txt
cat $DATA_DIR/dev.$FOLD.tsv  | tr '\t' ' '  | tr '  ' ' ' > dev.txt
cat $DATA_DIR/test.$FOLD.tsv  | tr '\t' ' '  | tr '  ' ' ' > test.txt
cat train.txt dev.txt test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > labels.txt


CUDA_VISIBLE_DEVICES=0 python3 run_ner.py \
--data_dir . \
--labels ./labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_eval \
--do_predict

done
