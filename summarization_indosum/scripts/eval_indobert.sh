#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python ../PreSumm/src_indo/train.py -task ext \
  -mode validate -batch_size 3000 -test_batch_size 500 \
  -bert_data_path ../data/indosum/presum_indobert1/ -log_file ../experiment/indobert1/log_val \
  -model_path  ../experiment/indobert1 -sep_optim true -use_interval true \
  -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 \
  -result_path ../experiment/indobert1/result -test_all -block_trigram false
