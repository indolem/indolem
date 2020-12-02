#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python ../PreSumm/src_multi/train.py -task ext \
    -mode validate -batch_size 3000 -test_batch_size 500 -bert_data_path ../data/presum_mbert1/ \
    -log_file ../experiment/mbert1/log_val -model_path  ../experiment/mbert1 -sep_optim true \
    -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 \
    -min_length 50 -result_path ../experiment/mbert1/result -test_all -uncased True \
    -block_trigram false
