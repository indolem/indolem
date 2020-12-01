#!/usr/bin/env python
# coding: utf-8


import json, glob, os, random
import re, tensorflow.keras, os
import pandas as pd, keras, io
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input,Dropout, Embedding, LSTM, Bidirectional, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['GOTO_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

args_parser = argparse.ArgumentParser()
args_parser.add_argument('--data_path', default='data/', help='path to all train/test/dev folds')
args_parser.add_argument('--max_word_len', type=int, default=100, help='maximum word allowed for 1 instance')
args_parser.add_argument('--max_nb_words', type=int, default=50000, help='maximum size of vocabulary')
args_parser.add_argument('--embedding_dim', type=int, default=300, help='embedding dimension of fasttext')
args_parser.add_argument('--num_class', type=int, default=1, help='number of class, we set 1 here because we use sigmoid')
args_parser.add_argument('--patience', type=int, default=20, help='patience count for early stopping')
args_parser.add_argument('--iterations', type=int, default=100, help='total epoch')
args_parser.add_argument('--batch_size', type=int, default=80, help='total batch size')
args_parser.add_argument('--fasttext_path', default='../cc.id.300.vec', help='path to indonesian fasttext')
args_parser.add_argument('--max_token_premise', type=int, default=100, help='maximum word for premise (can be more than 1 tweet')
args_parser.add_argument('--max_token_nextTw', type=int, default=30, help='maximum word for next tweet')

args = args_parser.parse_args()


def load_vectors(fname, word_index):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        if word_index.get(tokens[0],-1) != -1:
            data[tokens[0]] = list(map(float, tokens[1:]))
    return data


def preprocess_one(premise, nextTw, label):
    premise_subtokens = re.findall(r'\w+', premise.lower())
    if len(premise_subtokens) > args.max_token_premise:
        premise_subtokens = premise_subtokens[len(premise_subtokens)-args.max_token_premise:]

    nextTw_subtokens = re.findall(r'\w+', nextTw.lower())
    if len(nextTw_subtokens) > args.max_token_nextTw:
        nextTw_subtokens = nextTw_subtokens[:args.max_token_nextTw]
    return ' '.join(premise_subtokens), ' '.join(nextTw_subtokens), label


def preprocess(premises, nextTws, labels):
    assert len(premises) == len(nextTws) == len(labels)
    output_premise = []; output_nextTw = []; output_label = []
    for idx in range(len(premises)):
        a,b,c = preprocess_one(premises[idx], nextTws[idx], labels[idx])
        output_premise.append(a)
        output_nextTw.append(b)
        output_label.append(c)
    return output_premise, output_nextTw, np.asarray(output_label)


def model_with_fasttext(train_premise, train_nextTw, train_label, dev_premise, dev_nextTw, dev_label, test_premise, test_nextTw, test_label, tokenizer):
    word_index = tokenizer.word_index
    nb_words = min(args.max_nb_words, len(word_index))
    print('Total words in dict:', nb_words)
    embeddings = load_vectors(args.fasttext_path, word_index)
    embedding_matrix = np.zeros((nb_words + 1, args.embedding_dim
        ))
    for word, i in word_index.items():
        if i > args.max_nb_words:
            continue
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.random.normal(-4.2, 4.2, args.embedding_dim)
    
    print('Begin the training!')
    with tf.device('/gpu:0'):
        embedding_layer = Embedding(nb_words + 1, args.embedding_dim, weights=[embedding_matrix], 
                                input_length=args.max_word_len, trainable=False)
        
        premise = Input(shape=(args.max_token_premise,), dtype='int32')
        nextTw = Input(shape=(args.max_token_nextTw,), dtype='int32')
        embedded_premises = embedding_layer(premise)
        embedded_nextTws = embedding_layer(nextTw)
        lstm_premise = LSTM(units=200, activation='tanh', recurrent_activation='hard_sigmoid', 
                recurrent_regularizer=keras.regularizers.l2(0.2), return_sequences=False, dropout=0.3, recurrent_dropout=0.3)
        bilstm = Bidirectional(lstm_premise, merge_mode='concat')
        
        premise_vector = bilstm(embedded_premises)
        nextTw_vector = bilstm(embedded_nextTws)
        
        joined = premise_vector - nextTw_vector
        sign = Dense(args.num_class, activation='sigmoid')(joined)
        sent_model = Model([premise, nextTw], [sign])
        sent_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

        bestacc=0.0; patience = 0
        for i in range(args.iterations):
            if patience is args.patience:
                break
            sent_model.fit([train_premise, train_nextTw], [train_label], batch_size=args.batch_size, epochs=1, shuffle=True, verbose=True)
            prediction=sent_model.predict([dev_premise, dev_nextTw], batch_size=1000)
            batch_size = prediction.shape[0]
            predicted_label = np.reshape(prediction, (int(batch_size/4), 4))
            predicted_label = np.argmax(predicted_label,axis=1)
            gold_label = np.reshape(dev_label, (int(batch_size/4), 4))
            gold_label = np.argmax(gold_label,axis=1)
            
            accscore = accuracy_score(gold_label,predicted_label)
            if accscore > bestacc:
                print('Epoch ' + str(i) +' with dev acc: '+ str(accscore))
                bestacc = accscore
                sent_model.save('save.keras')
                patience = 0
            else:
                patience += 1
        sent_model = load_model('save.keras')
        prediction=sent_model.predict([test_premise, test_nextTw], batch_size=1000)
        batch_size = prediction.shape[0]
        predicted_label = np.reshape(prediction, (int(batch_size/4), 4))
        predicted_label = np.argmax(predicted_label,axis=1)
        gold_label = np.reshape(test_label, (int(batch_size/4), 4))
        gold_label = np.argmax(gold_label,axis=1)
    accscore = accuracy_score(gold_label,predicted_label)
    print('Test Acc: ',accscore)
    print('-----------------------------------------------------------------------------------')
    return accscore


def train_and_test_fasttext(trainset, devset, testset):
    train_premise, train_nextTw, train_label = trainset
    dev_premise, dev_nextTw, dev_label = devset
    test_premise, test_nextTw, test_label = testset

    tokenizer = Tokenizer(num_words=args.max_nb_words, lower=True)
    tokenizer.fit_on_texts(train_premise + train_nextTw)
    
    train_premise = tokenizer.texts_to_sequences(train_premise)
    train_nextTw = tokenizer.texts_to_sequences(train_nextTw)
    dev_premise = tokenizer.texts_to_sequences(dev_premise)
    dev_nextTw = tokenizer.texts_to_sequences(dev_nextTw)
    test_premise = tokenizer.texts_to_sequences(test_premise)
    test_nextTw = tokenizer.texts_to_sequences(test_nextTw)
    
    train_premise = sequence.pad_sequences(train_premise, maxlen=args.max_token_premise, padding='post')
    train_nextTw = sequence.pad_sequences(train_nextTw, maxlen=args.max_token_nextTw, padding='post')
    dev_premise = sequence.pad_sequences(dev_premise, maxlen=args.max_token_premise, padding='post')
    dev_nextTw = sequence.pad_sequences(dev_nextTw, maxlen=args.max_token_nextTw, padding='post')
    test_premise = sequence.pad_sequences(test_premise, maxlen=args.max_token_premise, padding='post')
    test_nextTw = sequence.pad_sequences(test_nextTw, maxlen=args.max_token_nextTw, padding='post')
    
    return model_with_fasttext(train_premise, train_nextTw, train_label, \
                                dev_premise, dev_nextTw, dev_label,\
                                test_premise, test_nextTw, test_label,tokenizer)

def clean(s):
    s = re.findall(r'[\w,\w+!,\w?,\w.]+', s.lower())
    return ' '.join(s)


def read_data(fname):
    premise = [] #will be 4 times
    nextTw = []
    label = []
    data=json.load(open(fname,'r'))
    for datum in data:
        for key, option in datum['next_tweet']:
            c = '. '.join(datum['tweets'])
            premise.append(clean(c))
            nextTw.append(clean(option))
            label.append(key)
    return premise, nextTw, label


trainset = read_data(args.data_path+'train.json')
devset = read_data(args.data_path+'dev.json')
testset = read_data(args.data_path+'test.json')
train_dataset = preprocess(trainset[0], trainset[1], trainset[2])
dev_dataset = preprocess(devset[0], devset[1], devset[2])
test_dataset = preprocess(testset[0], testset[1], testset[2])
train_and_test_fasttext(train_dataset, dev_dataset, test_dataset)
