#!/usr/bin/env python
# coding: utf-8


import json, glob, os, random
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from itertools import permutations
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)
lang2model = { 'id': 'indolem/indobert-base-uncased',
               'multi': 'bert-base-multilingual-uncased',
               'my': 'huseinzol05/bert-base-bahasa-cased' }
lang2pad = {'id': 0, 'multi': 0, 'my': 5}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

class BertData():
    def __init__(self, args):
        self.MAX_TOKEN_TWEET = args.max_token_tweet
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'

        if args.bert_lang == 'id' or args.bert_lang == 'multi':
            self.tokenizer = BertTokenizer.from_pretrained(lang2model[args.bert_lang], do_lower_case=True)
            self.sep_vid = self.tokenizer.vocab[self.sep_token]
            self.cls_vid = self.tokenizer.vocab[self.cls_token]
            self.pad_vid = self.tokenizer.vocab[self.pad_token]
        elif args.bert_lang == 'my':
            self.tokenizer = AlbertTokenizer.from_pretrained(lang2model[args.bert_lang],
                    unk_token = '[UNK]', pad_token='[PAD]', do_lower_case=False)
            self.sep_vid = self.tokenizer.convert_tokens_to_ids(self.sep_token)
            self.cls_vid = self.tokenizer.convert_tokens_to_ids(self.cls_token)
            self.pad_vid = self.tokenizer.convert_tokens_to_ids(self.pad_token)


    def preprocess_one(self, tweet, label):
        tweet_subtoken_ids = [self.tokenizer.convert_tokens_to_ids([self.cls_token]+self.tokenizer.tokenize(t)[:self.MAX_TOKEN_TWEET]+[self.sep_token]) for t in tweet]
        tmp = []
        for arr in tweet_subtoken_ids:
            tmp += arr
        tweet_subtoken_ids = tmp
        cls_id = []; segment_id = []
        cls_mask = []
        flip=False
        for i, token in enumerate(tweet_subtoken_ids):
            if token == self.cls_vid:
                cls_id.append(i)
                cls_mask.append(1)
                flip = not flip
            if flip:
                segment_id.append(0)
            else:
                segment_id.append(1)
        idx=len(cls_id)
        while idx < 5:
            cls_id.append(0)
            cls_mask.append(0)
            label.append(5)
            idx+=1
        return tweet_subtoken_ids, segment_id, cls_id, cls_mask, label
    
    def preprocess(self, tweets, labels):
        assert len(tweets) == len(labels)
        output = []
        for idx in range(len(tweets)):
            output.append(self.preprocess_one(tweets[idx], labels[idx]))
        return output


class Batch():
    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data
    
    # do padding here
    def __init__(self, data, idx, batch_size, device):
        PAD_ID=lang2pad[args.bert_lang]
        cur_batch = data[idx:idx+batch_size]
        src = torch.tensor(self._pad([x[0] for x in cur_batch], PAD_ID))
        seg = torch.tensor(self._pad([x[1] for x in cur_batch], PAD_ID))
        cls_id = torch.tensor([x[2] for x in cur_batch])
        cls_mask = torch.tensor([x[3] for x in cur_batch])
        label = torch.tensor([x[4] for x in cur_batch])
        mask_src = 0 + (src!=PAD_ID)
        
        self.src = src.to(device)
        self.seg= seg.to(device)
        self.label = label.to(device)
        self.mask_src = mask_src.to(device)
        self.cls_id = cls_id.to(device)
        self.cls_mask = cls_mask.to(device)

    def get(self):
        return self.src, self.seg, self.label, self.mask_src, self.cls_id, self.cls_mask


class Model(nn.Module):
    def __init__(self, args, device):
        super(Model, self).__init__()
        self.args = args
        self.device = device
        self.bert = BertModel.from_pretrained(lang2model[args.bert_lang])
        self.linear = nn.Linear(self.bert.config.hidden_size, 5)
        self.dropout = nn.Dropout(0.2)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=5, reduction='sum')
    
    def forward(self, src, seg, mask_src, cls_id, cls_mask):
        batch_size = src.shape[0]
        top_vec, _ = self.bert(input_ids=src, token_type_ids=seg, attention_mask=mask_src)
        
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), cls_id] #batch_size * 5 * dim
        sents_vec = sents_vec * cls_mask[:, :, None].float()
        final_rep = self.dropout(sents_vec)
        conclusion = self.linear(final_rep) #batch_size * 5 * 5
        return conclusion #batch_size * 5 * 5
    
    def get_loss(self, src, seg, label, mask_src, cls_id, cls_mask):
        output = self.forward(src, seg, mask_src, cls_id, cls_mask)
        return self.loss(output.view(-1,5), label.view(-1))

    def compute(self, matrix, length):
        ids = list(permutations(np.arange(length),length))
        ids = [list(i) for i in ids]
        maxs = []; max_score = 0
        for x in ids:
            score = 0.0
            for j, i in enumerate(x):
                score += matrix[j][i]
            if score > max_score:
                max_score = score
                maxs = x
        return maxs

    def predict_cor(self, src, seg, mask_src, label, cls_id, cls_mask):
        output = self.forward(src, seg, mask_src, cls_id, cls_mask)
        batch_size = output.shape[0]
        cors = []
        for idx in range(batch_size):
            limit = cls_mask[idx].sum()
            cur_output = output[idx][:limit]
            cur_prediction = torch.nn.Softmax(dim=-1)(cur_output.masked_fill(cls_mask[idx]==0, -np.inf))
            
            pred_rank = self.compute(cur_prediction.data.cpu().tolist(), limit.item())
            gold_rank = label[idx].data.cpu().tolist()[:limit]
            coef, _ = spearmanr(pred_rank, gold_rank)
            cors.append(coef)
        return cors


def prediction(dataset, model, args):
    rank_cors = []
    model.eval()
    for j in range(0, len(dataset), args.batch_size):
        src, seg, label, mask_src, cls_id, cls_mask = Batch(dataset, j, args.batch_size, args.device).get()
        cors = model.predict_cor(src, seg, mask_src, label, cls_id, cls_mask)
        rank_cors += cors
    return np.mean(rank_cors)


def read_data(fname):
    tweets = []
    labels = []
    data=json.load(open(fname,'r'))
    for datum in data:
        tweets.append(datum['tweets'])
        labels.append(datum['order'])
    return tweets, labels


def train(args, train_dataset, dev_dataset, test_dataset, model):
    """ Train the model """
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    t_total = len(train_dataset) // args.batch_size * args.num_train_epochs
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Warming up = %d", args.warmup_steps)
    logger.info("  Patience  = %d", args.patience)

    # Added here for reproductibility
    set_seed(args)
    tr_loss = 0.0
    global_step = 1
    best_acc_dev = 0
    best_acc_test = 0
    cur_patience = 0
    for i in range(int(args.num_train_epochs)):
        random.shuffle(train_dataset)
        epoch_loss = 0.0
        epoch_step = 1
        for j in range(0, len(train_dataset), args.batch_size):
            src, seg, label, mask_src, cls_id, cls_mask = Batch(train_dataset, j, args.batch_size, args.device).get()
            model.train()
            loss = model.get_loss(src, seg, label, mask_src, cls_id, cls_mask)
            loss = loss.sum()/args.batch_size
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            loss.backward()

            tr_loss += loss.item()
            epoch_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1
            epoch_step += 1
        logger.info("Finish epoch = %s, loss_epoch = %s", i+1, epoch_loss/epoch_step)
        dev_acc = prediction(dev_dataset, model, args)
        if dev_acc > best_acc_dev:
            best_acc_dev = dev_acc
            test_acc = prediction(test_dataset, model, args)
            best_acc_test = test_acc
            cur_patience = 0
            logger.info("Better, BEST RankCorr in DEV = %s & BEST RankCorr in test = %s.", best_acc_dev, best_acc_test)
        else:
            cur_patience += 1
            if cur_patience == args.patience:
                logger.info("Early Stopping Not Better, BEST RankCorr in DEV = %s & BEST RankCorr in test = %s.", best_acc_dev, best_acc_test)
                break
            else:
                logger.info("Not Better, BEST RankCorr in DEV = %s & BEST RankCorr in test = %s.", best_acc_dev, best_acc_test)

    return global_step, tr_loss / global_step, best_acc_dev, best_acc_test


args_parser = argparse.ArgumentParser()
args_parser.add_argument('--data_path', default='data/', help='path to all train/test/dev folds')
args_parser.add_argument('--bert_lang', default='id', choices=['id', 'multi', 'my'], help='select one of language')
args_parser.add_argument('--max_token_tweet', type=int, default=50, help='maximum token (subwords) allowed for 1 tweet')
args_parser.add_argument('--batch_size', type=int, default=20, help='batch size')
args_parser.add_argument('--learning_rate', type=float, default=5e-5, help='learning rate')
args_parser.add_argument('--weight_decay', type=int, default=0, help='weight decay')
args_parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='adam epsilon')
args_parser.add_argument('--max_grad_norm', type=float, default=1.0)
args_parser.add_argument('--num_train_epochs', type=int, default=20, help='total epoch')
args_parser.add_argument('--warmup_steps', type=int, default=532, help='warmup_steps, the default value is 10% of total steps')
args_parser.add_argument('--logging_steps', type=int, default=200, help='report stats every certain steps')
args_parser.add_argument('--seed', type=int, default=2020)
args_parser.add_argument('--local_rank', type=int, default=-1)
args_parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')
args_parser.add_argument('--no_cuda', default=False)
args = args_parser.parse_args()


# Setup CUDA, GPU & distributed training
if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
else: # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.n_gpu = 1
args.device = device

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
)

set_seed(args)

if args.local_rank not in [-1, 0]:
    torch.distributed.barrier()
if args.local_rank == 0:
    torch.distributed.barrier()

bertdata = BertData(args)

dev_rankCorrs = 0.0
test_rankCorrs = 0.0
for idx in range(5):
    trainset = read_data(args.data_path+'train'+str(idx)+'.json')
    devset = read_data(args.data_path+'dev'+str(idx)+'.json')
    testset = read_data(args.data_path+'test'+str(idx)+'.json')
    model = Model(args, device)
    model.to(args.device)
    train_dataset = bertdata.preprocess(trainset[0], trainset[1])
    dev_dataset = bertdata.preprocess(devset[0], devset[1])
    test_dataset = bertdata.preprocess(testset[0], testset[1])
    
    global_step, tr_loss, best_rankCorr_dev, best_rankCorr_test = train(args, train_dataset, dev_dataset, test_dataset, model)
    dev_rankCorrs += best_rankCorr_dev
    test_rankCorrs += best_rankCorr_test

print('End of Training 5-fold')
print('Dev set RankCorr', dev_rankCorrs/5.0)
print('Test set RankCorr', test_rankCorrs/5.0)


