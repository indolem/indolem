#!/usr/bin/env python
# coding: utf-8


import json, glob, os, random
import re, tensorflow.keras, os
import pandas as pd, keras, io
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input,Dropout, Embedding, LSTM, Bidirectional, Activation, Masking
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from bpe import Encoder
from tensorflow.keras.backend import reshape
import keras.backend as K
from scipy.stats import spearmanr
from itertools import permutations
import argparse


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['GOTO_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

args_parser = argparse.ArgumentParser()
args_parser.add_argument('--data_path', default='data/', help='path to all train/test/dev folds')
args_parser.add_argument('--max_nb_words', type=int, default=50000, help='maximum size of vocabulary')
args_parser.add_argument('--embedding_dim', type=int, default=300, help='embedding dimension of fasttext')
args_parser.add_argument('--num_class', type=int, default=1, help='number of class, we set 1 here because we use sigmoid')
args_parser.add_argument('--patience', type=int, default=20, help='patience count for early stopping')
args_parser.add_argument('--iterations', type=int, default=100, help='total epoch')
args_parser.add_argument('--batch_size', type=int, default=200, help='total batch size')
args_parser.add_argument('--fasttext_path', default='../cc.id.300.vec', help='path to indonesian fasttext')
args_parser.add_argument('--max_token_tweet', type=int, default=30, help='maximum word allowed for 1 tweet')
args_parser.add_argument('--max_tweet', type=int, default=5, help='maximum number of tweet in 1 instance')

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

def preprocess_one(tweet):
    result = []
    for tw in tweet:
        tw_subtokens = re.findall(r'\w+', tw.lower())
        if len(tw_subtokens) > args.max_token_tweet:
            tw_subtokens = tw_subtokens[:args.max_token_tweet]
        result.append(' '.join(tw_subtokens))
    return result

def preprocess(tweets, denominators, masks, orders):
    assert len(tweets) == len(orders) == len(masks)
    output_tweets = []; 
    for idx in range(len(tweets)):
        tweet = preprocess_one(tweets[idx])
        output_tweets.append(tweet)
    return output_tweets, denominators, masks, orders

def get_best_rank(matrix, length):
    length = int(length)
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

def compute_corr(prob_distrib, mask, order):
    prob_distrib = prob_distrib.numpy()
    corrs = []
    for idx in range(prob_distrib.shape[0]):
        limit = int(mask[idx].sum())
        predict_order = get_best_rank(prob_distrib[idx], limit)
        coef, _ = spearmanr(predict_order, order[idx][:limit])
        corrs.append(coef)
    return np.mean(corrs)
        
def customLoss(label, logits):
    gt = tf.reshape(label, [-1])
    raw_prediction = reshape(logits,(-1, 5))
    indices = tf.squeeze(tf.where(tf.not_equal(gt, 5)),1)
    gt = tf.cast(tf.gather(gt,indices), tf.int32)
    prediction = tf.gather(raw_prediction, indices)
    loss=tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction,labels=gt,name="entropy")))
    return loss

def model_with_fasttext(train_tweet, dev_tweet, test_tweet, \
        train_mask, dev_mask, test_mask,
        train_order, dev_order, test_order, 
        train_denom, dev_denom, test_denom, tokenizer):
    word_index = tokenizer.word_index
    nb_words = min(args.max_nb_words, len(word_index))
    print('Total words in dict:', nb_words)
    embeddings = load_vectors(args.fasttext_path, word_index)
    embedding_matrix = np.zeros((nb_words + 1, args.embedding_dim))
    for word, i in word_index.items():
        if i > args.max_nb_words:
            continue
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.random.normal(-4.2, 4.2, args.embedding_dim)
    print('Finish to read Fast Text Embedding')
    
    print('Begin the training!')
    best_rankCorr_dev = 0.0; best_rankCorr_test = 0.0;
    with tf.device('/gpu:0'):
        embedding_layer = Embedding(nb_words + 1, args.embedding_dim, weights=[embedding_matrix], 
                                input_length=args.max_token_tweet, trainable=True, mask_zero=True)
        tweet = Input(shape=(5, args.max_token_tweet,), dtype='int32', name='tweet')
        denom = Input(shape=(5,), dtype='float32', name='denom')

        embedded_tweet = embedding_layer(tweet) #batch x 5 x #token x #hidden
        mask_tweet = Masking(mask_value=0)(tweet)
        mask_tweet = tf.cast(mask_tweet, tf.float32)

        lstm1 = LSTM(units=200, activation='tanh', recurrent_activation='hard_sigmoid', 
                recurrent_regularizer=keras.regularizers.l2(0.2), return_sequences=True, dropout=0.3, recurrent_dropout=0.3)
        bilstm1 = Bidirectional(lstm1, merge_mode='concat')
        
        batch_size, num_chat, num_token, hidden_size = embedded_tweet.shape
        embedded_tweet = reshape(embedded_tweet, shape=(-1, num_token, hidden_size))
        
        mask_tweet = reshape(mask_tweet, shape=(-1, args.max_token_tweet, 1)) #batch * 5 x #word x 1
        tweet_vector = bilstm1(embedded_tweet)  * mask_tweet #batch * 5 x # word x #hidden
        # averaging after bilstm
        tweet_vector = tf.keras.backend.sum(tweet_vector, axis=1, keepdims=False) #batch * 5 x #hidden
        tweet_vector = tweet_vector /reshape(denom, shape=(-1, 1))
        tweet_vector = reshape(tweet_vector, shape=(-1, args.max_tweet, 400)) #batch x 5 x #hidden

        lstm2 = LSTM(units=200, activation='tanh', recurrent_activation='hard_sigmoid', 
                recurrent_regularizer=keras.regularizers.l2(0.2), return_sequences=True, dropout=0.3, recurrent_dropout=0.3)
        bilstm2 = Bidirectional(lstm2, merge_mode='concat')
        tweet_vector2 = bilstm2(tweet_vector)  #batch x 5 x #hidden

        mlp = Dense(args.max_tweet)
        output = mlp(tweet_vector2) #batch x 5 x 5
        # activation for output and softmax is handled by customLoss
        sent_model = Model([tweet, denom], [output])
        sent_model.compile(loss=customLoss, optimizer='adam')

        best_rankCorr_dev=0.0; patience = 0
        for i in range(args.iterations):
            if patience == args.patience:
                break
            sent_model.fit([train_tweet, train_denom], [train_order], batch_size=args.batch_size, epochs=1, shuffle=True, verbose=True)
            prob_distrib = sent_model.predict([dev_tweet, dev_denom], batch_size=1000)
            tmp = tf.where(dev_mask!=0, 1-dev_mask, -1e50)
            prob_distrib += reshape(tmp, [-1,1,5])
            prob_distrib = tf.keras.activations.softmax(prob_distrib, axis=-1)
            prob_distrib *= reshape(dev_mask, [-1,5,1]) # not necessary
            corr_score = compute_corr(prob_distrib, dev_mask, dev_order)
            
            if corr_score > best_rankCorr_dev:
                print('Epoch ' + str(i) +' has better dev rank corr: '+ str(corr_score))
                best_rankCorr_dev = corr_score
                patience = 0
                #Predict Test
                prob_distrib = sent_model.predict([test_tweet, test_denom], batch_size=1000)
                tmp = tf.where(test_mask!=0, 1-test_mask, -1e50)
                prob_distrib += reshape(tmp, [-1,1,5])
                prob_distrib = tf.keras.activations.softmax(prob_distrib, axis=-1)
                prob_distrib *= reshape(test_mask, [-1,5,1]) #not necessary
                best_rankCorr_test = compute_corr(prob_distrib, test_mask, test_order)
                print('Epoch ' + str(i) +' test rank corr: '+ str(best_rankCorr_test))
            else:
                patience += 1
    
    print('Dev Rank Corr:', best_rankCorr_dev,'Test Rank Corr:', best_rankCorr_test)
    print('-----------------------------------------------------------------------------------')
    return best_rankCorr_dev, best_rankCorr_test


def tokenize(tokenizer, tweets, denoms):
    tokenized_tweets = []
    for idx, tweet in enumerate(tweets):
        tokenized_tweet = tokenizer.texts_to_sequences(tweet)
        for idy in range(5):
            if len(tokenized_tweet[idy]) != 0 and len(tokenized_tweet[idy]) < args.max_token_tweet:
                denoms[idx][idy] = len(tokenized_tweet[idy])
        tokenized_tweets.append(tokenized_tweet)
    return tokenized_tweets, denoms

def train_and_test_fasttext(trainset, devset, testset):
    train_tweet, train_denom, train_mask, train_order = trainset
    dev_tweet, dev_denom, dev_mask, dev_order = devset
    test_tweet, test_denom, test_mask, test_order = testset
    
    fulldata=[]
    for tweets in train_tweet:
        for tweet in tweets:
            fulldata.append(tweet)
    fulldata=np.array(fulldata)
    tokenizer = Tokenizer(num_words=args.max_nb_words, lower=True)
    tokenizer.fit_on_texts(fulldata)
    
    train_tweet, train_denom = tokenize(tokenizer, train_tweet, train_denom)
    dev_tweet, dev_denom = tokenize(tokenizer, dev_tweet, dev_denom)
    test_tweet, test_denom = tokenize(tokenizer, test_tweet, test_denom)
    train_tweet = [sequence.pad_sequences(tweet, maxlen=args.max_token_tweet, padding='post') for tweet in train_tweet]
    dev_tweet = [sequence.pad_sequences(tweet, maxlen=args.max_token_tweet, padding='post') for tweet in dev_tweet]
    test_tweet = [sequence.pad_sequences(chat, maxlen=args.max_token_tweet, padding='post') for chat in test_tweet]
    
    train_tweet = np.array(train_tweet)
    test_tweet = np.array(test_tweet)
    dev_tweet = np.array(dev_tweet)
    train_order = np.array(train_order)
    test_order = np.array(test_order)
    dev_order = np.array(dev_order)
    train_mask = np.array(train_mask)
    test_mask = np.array(test_mask)
    dev_mask = np.array(dev_mask)
    train_denom = np.array(train_denom, dtype=np.float)
    test_denom = np.array(test_denom, dtype=np.float)
    dev_denom = np.array(dev_denom, dtype=np.float)

    return model_with_fasttext(train_tweet, dev_tweet, test_tweet, train_mask, dev_mask, test_mask, train_order, dev_order, test_order, train_denom, dev_denom, test_denom, tokenizer)


def read_data(fname):
    tweets = []
    masks = []
    denominators = []
    orders = []
    data=json.load(open(fname,'r'))
    for datum in data:
        mask = [0.0] * 5
        denom = [args.max_token_tweet] * 5
        for idx in range(len(datum['tweets'])):
            mask[idx] = 1.0
        masks.append(mask)
        while(len(datum['tweets'])<5):
            datum['tweets'].append('')
            datum['order'].append(5)
        denominators.append(denom)
        tweets.append(datum['tweets'])
        orders.append(datum['order'])
    return tweets, denominators, masks, orders


print('Experiment with 5-fold Cross Validation')
dev_rankCorrs = 0.0
test_rankCorrs = 0.0
for idx in range(5):
    trainset = read_data(args.data_path+'train'+str(idx)+'.json')
    devset = read_data(args.data_path+'dev'+str(idx)+'.json')
    testset = read_data(args.data_path+'test'+str(idx)+'.json')
    train_dataset = preprocess(trainset[0], trainset[1], trainset[2], trainset[3])
    dev_dataset = preprocess(devset[0], devset[1], devset[2], devset[3])
    test_dataset = preprocess(testset[0], testset[1], testset[2], testset[3])
    
    dev_score, test_score = train_and_test_fasttext(train_dataset, dev_dataset, test_dataset)
    dev_rankCorrs += dev_score
    test_rankCorrs += test_score

print('End of Training 5-fold')
print('Dev set RankCorr', dev_rankCorrs/5.0)
print('Test set RankCorr', test_rankCorrs/5.0)


