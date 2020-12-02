#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2 python ../PreSumm/src_multi/train.py -task ext \
    -mode train -bert_data_path ../data/presum_mbert1/ -model_path ../experiment/mbert1 \
    -ext_dropout 0.1 -lr 2e-3 -visible_gpus 0,1,2 -report_every 50 -save_checkpoint_steps 2500 \
    -batch_size 3000 -train_steps 20000 -accum_count 2 \
    -log_file ../experiment/mbert1/log -warmup_steps 4000 -max_pos 512
