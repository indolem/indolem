#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 python -u ../parsing_conllu.py --mode train --config ../configs/parsing/bertBiaffine.json --num_epochs 400 --batch_size 32 \
 --opt adam --learning_rate 0.001 --lr_decay 0.999995 --beta1 0.9 --beta2 0.9 --eps 1e-4 --grad_clip 5.0 \
 --loss_type token --warmup_steps 40 --reset 20 --weight_decay 0.0 --unk_replace 0.5 \
 --punctuation '.' '``' "''" ':' ',' '?' '!' \
 --word_embedding bert --word_path "../../UD_Indonesian_GSD/data/malaybert" --char_embedding random \
 --word2index_path "../../UD_Indonesian_GSD/data/word2index.json" \
 --train "../../UD_Indonesian_GSD/data/train.conllu" \
 --dev "../../UD_Indonesian_GSD/data/dev.conllu" \
 --test "../../UD_Indonesian_GSD/data/test.conllu" \
 --model_path "models/malaybertGSD/" \
 --normalize_digits 0
