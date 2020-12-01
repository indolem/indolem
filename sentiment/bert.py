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
from transformers import BertTokenizer, AlbertTokenizer, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup

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
        self.MAX_TOKEN = args.max_token
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

    def preprocess_one(self, src_txt, label):
        src_subtokens = [self.cls_token] + self.tokenizer.tokenize(src_txt) + [self.sep_token]        
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        
        if len(src_subtoken_idxs) > self.MAX_TOKEN:
            src_subtoken_idxs = src_subtoken_idxs[:self.MAX_TOKEN]
            src_subtoken_idxs[-1] = self.sep_vid
        else:
            src_subtoken_idxs += [self.pad_vid] * (self.MAX_TOKEN-len(src_subtoken_idxs))
        segments_ids = [0] * len(src_subtoken_idxs)
        assert len(src_subtoken_idxs) == len(segments_ids)
        return src_subtoken_idxs, segments_ids, label
    
    def preprocess(self, src_txts, labels):
        assert len(src_txts) == len(labels)
        output = []
        for idx in range(len(src_txts)):
            output.append(self.preprocess_one(src_txts[idx], labels[idx]))
        return output


class Batch():
    def __init__(self, data, idx, batch_size, device):
        cur_batch = data[idx:idx+batch_size]
        src = torch.tensor([x[0] for x in cur_batch])
        seg = torch.tensor([x[1] for x in cur_batch])
        label = torch.tensor([x[2] for x in cur_batch])
        mask_src = 0 + (src!=lang2pad[args.bert_lang])
        
        self.src = src.to(device)
        self.seg= seg.to(device)
        self.label = label.to(device)
        self.mask_src = mask_src.to(device)

    def get(self):
        return self.src, self.seg, self.label, self.mask_src


class Model(nn.Module):
    def __init__(self, args, device):
        super(Model, self).__init__()
        self.args = args
        self.device = device
        self.bert = BertModel.from_pretrained(lang2model[args.bert_lang])
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        self.loss = torch.nn.BCELoss(reduction='none') 

    def forward(self, src, seg, mask_src):
        top_vec, _ = self.bert(input_ids=src, token_type_ids=seg, attention_mask=mask_src)
        top_vec = self.dropout(top_vec)
        top_vec *= mask_src.unsqueeze(dim=-1).float()
        top_vec = torch.sum(top_vec, dim=1) / mask_src.sum(dim=-1).float().unsqueeze(-1)
        conclusion = self.linear(top_vec).squeeze()
        return self.sigmoid(conclusion)
    
    def get_loss(self, src, seg, label, mask_src):
        output = self.forward(src, seg, mask_src)
        return self.loss(output, label.float())

    def predict(self, src, seg, mask_src):
        output = self.forward(src, seg, mask_src)
        prediction = output.cpu().data.numpy() > 0.5
        if type (prediction) == np.bool_:
            return [int(prediction)]
        return [int(x) for x in prediction]


def prediction(dataset, model, args):
    preds = []
    golds = []
    model.eval()
    for j in range(0, len(dataset), args.batch_size):
        src, seg, label, mask_src = Batch(dataset, j, args.batch_size, args.device).get()
        preds += model.predict(src, seg, mask_src)
        golds += label.cpu().data.numpy().tolist()
    return f1_score(golds, preds), accuracy_score(golds, preds)


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
    best_f1_dev = 0
    best_f1_test = 0
    cur_patience = 0
    for i in range(int(args.num_train_epochs)):
        random.shuffle(train_dataset)
        epoch_loss = 0.0
        for j in range(0, len(train_dataset), args.batch_size):
            src, seg, label, mask_src = Batch(train_dataset, j, args.batch_size, args.device).get()
            model.train()
            loss = model.get_loss(src, seg, label, mask_src)
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
        
        logger.info("Finish epoch = %s, loss_epoch = %s", i+1, epoch_loss/global_step)
        dev_f1, dev_acc = prediction(dev_dataset, model, args)
        if dev_f1 > best_f1_dev:
            best_f1_dev = dev_f1
            test_f1, test_acc = prediction(test_dataset, model, args)
            best_f1_test = test_f1
            cur_patience = 0
            logger.info("Better, BEST F1 in DEV = %s & BEST F1 in test = %s.", best_f1_dev, best_f1_test)
        else:
            cur_patience += 1
            if cur_patience == args.patience:
                logger.info("Early Stopping Not Better, BEST F1 in DEV = %s & BEST F1 in test = %s.", best_f1_dev, best_f1_test)
                break
            else:
                logger.info("Not Better, BEST F1 in DEV = %s & BEST F1 in test = %s.", best_f1_dev, best_f1_test)

    return global_step, tr_loss / global_step, best_f1_dev, best_f1_test


args_parser = argparse.ArgumentParser()
args_parser.add_argument('--data_path', default='data/', help='path to all train/test/dev folds')
args_parser.add_argument('--bert_lang', default='id', choices=['id', 'multi', 'my'], help='select one of language')
args_parser.add_argument('--max_token', type=int, default=200, help='maximum token allowed for 1 instance')
args_parser.add_argument('--batch_size', type=int, default=30, help='batch size')
args_parser.add_argument('--learning_rate', type=float, default=5e-5, help='learning rate')
args_parser.add_argument('--weight_decay', type=int, default=0, help='weight decay')
args_parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='adam epsilon')
args_parser.add_argument('--max_grad_norm', type=float, default=1.0)
args_parser.add_argument('--num_train_epochs', type=int, default=20, help='total epoch')
args_parser.add_argument('--warmup_steps', type=int, default=242, help='warmup_steps, the default value is 10% of total steps')
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

dev_f1s = 0.0
test_f1s = 0.0
for idx in range(5):
    trainset = pd.read_csv(args.data_path+'train'+str(idx)+'.csv')
    devset = pd.read_csv(args.data_path+'dev'+str(idx)+'.csv')
    testset = pd.read_csv(args.data_path+'test'+str(idx)+'.csv')
    xtrain, ytrain = list(trainset['sentence']), list(trainset['sentiment'])
    xdev, ydev = list(devset['sentence']), list(devset['sentiment'])
    xtest, ytest = list(testset['sentence']), list(testset['sentiment'])
    model = Model(args, device)
    model.to(args.device)
    train_dataset = bertdata.preprocess(xtrain, ytrain)
    dev_dataset = bertdata.preprocess(xdev, ydev)
    test_dataset = bertdata.preprocess(xtest, ytest)
    
    global_step, tr_loss, best_f1_dev, best_f1_test = train(args, train_dataset, dev_dataset, test_dataset, model)
    dev_f1s += best_f1_dev
    test_f1s += best_f1_test

print('End of Training 5-fold')
print('Dev set F1', dev_f1s/5.0)
print('Test set F1', test_f1s/5.0)




