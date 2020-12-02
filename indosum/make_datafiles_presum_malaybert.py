import sys
import os
import shutil
import json, glob
import torch
from transformers import AlbertTokenizer

SHARD_SIZE = 2000
MIN_SRC_NSENTS = 3
MAX_SRC_NSENTS = 100
MIN_SRC_NTOKENS_PER_SENT = 5
MAX_SRC_NTOKENS_PER_SENT = 200
MIN_TGT_NTOKENS = 5
MAX_TGT_NTOKENS = 500
USE_BERT_BASIC_TOKENIZER = False

main_path = 'data/'

class BertData():
    def __init__(self):
        self.tokenizer = AlbertTokenizer.from_pretrained('huseinzol05/bert-base-bahasa-cased', 
                unk_token = '[UNK]', pad_token='[PAD]', do_lower_case=False)
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.bos_token = '<s>' #beg of summary
        self.eos_token = '</s>' # end of summary
        self.mid_token = '[EOP]' #segment between sentences within a summary
        self.sep_vid = self.tokenizer.convert_tokens_to_ids(self.sep_token)
        self.cls_vid = self.tokenizer.convert_tokens_to_ids(self.cls_token)
        self.pad_vid = self.tokenizer.convert_tokens_to_ids(self.pad_token)
        self.bos_vid = self.tokenizer.convert_tokens_to_ids(self.bos_token)
        self.eos_vid = self.tokenizer.convert_tokens_to_ids(self.eos_token)
        self.mid_vid = self.tokenizer.convert_tokens_to_ids(self.mid_token)
    
    def preprocess(self, src, tgt, sent_labels, use_bert_basic_tokenizer=False, is_test=False):

        original_src_txt = [' '.join(s) for s in src]

        idxs = [i for i, s in enumerate(src) if (len(s) > MIN_SRC_NTOKENS_PER_SENT)]

        _sent_labels = [0] * len(src)
        for l in sent_labels:
            _sent_labels[l] = 1

        src = [src[i][:MAX_SRC_NTOKENS_PER_SENT] for i in idxs]
        sent_labels = [_sent_labels[i] for i in idxs]
        src = src[:MAX_SRC_NSENTS]
        sent_labels = sent_labels[:MAX_SRC_NSENTS]

        if len(src) < MIN_SRC_NSENTS:
            return None

        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)

        src_subtokens = self.tokenizer.tokenize(text)

        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]
        
        QOS = ' [EOP] '
        tgt_subtokens_str = QOS.join([' '.join(self.tokenizer.tokenize(' '.join(tt))) for tt in tgt])
        tgt_subtoken = tgt_subtokens_str.split()[:MAX_TGT_NTOKENS]
        if len(tgt_subtoken) < MIN_TGT_NTOKENS:
            return None
        
        tgt_subtoken_idxs = [self.bos_vid] + self.tokenizer.convert_tokens_to_ids(tgt_subtoken) + [self.eos_vid]

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]
        return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt

def read(fname):
    data = []
    for line in open(fname, 'r').readlines():
        datum = json.loads(line)
        label = []; src= []; tgt = datum['summary']
        for idx in range(len(datum['paragraphs'])):
            for idy in range(len(datum['paragraphs'][idx])):
                src.append(datum['paragraphs'][idx][idy])
                label.append(datum['gold_labels'][idx][idy])
        label = [idx for idx, x in enumerate(label) if x]
        data.append((src, tgt, label))
    return data


def format_to_bert(path, data_path):
    bert = BertData()
    p_ct = 0
    dataset = []
    data = read(path)
    for datum in data:
        #process
        source, tgt, sent_labels = datum
        b_data = bert.preprocess(source, tgt, sent_labels)
        if (b_data is None):
            continue
        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                       "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt}
        dataset.append(b_data_dict)
        if len(dataset) >= SHARD_SIZE:
            pt_file = data_path + "{:s}.{:d}.bert.pt".format(path.split('/')[-1].split('.')[0], p_ct)
            torch.save(dataset, pt_file)
            dataset = []
            p_ct += 1
    if len(dataset) > 0:
        pt_file = data_path + "{:s}.{:d}.bert.pt".format(path.split('/')[-1].split('.')[0], p_ct)
        torch.save(dataset, pt_file)
        dataset = []
        p_ct += 1


for i in [1,2,3,4,5]:
    data_path = main_path + 'presum_malaybert'+str(i)+'/'
    print('Create ', data_path)
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    os.makedirs(data_path)
    format_to_bert(main_path + 'train.0'+str(i)+'.jsonl', data_path)
    format_to_bert(main_path + 'dev.0'+str(i)+'.jsonl', data_path)
    format_to_bert(main_path + 'test.0'+str(i)+'.jsonl', data_path)
