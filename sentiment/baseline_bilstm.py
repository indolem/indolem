#!/usr/bin/env python
# coding: utf-8


import re, tensorflow.keras, os
import pandas as pd, keras, io
import numpy as np
from tensorflow.keras.layers import Dense, Input, Dropout, Embedding, LSTM, Bidirectional, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from sklearn.metrics import f1_score,accuracy_score
import tensorflow as tf
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['GOTO_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

args_parser = argparse.ArgumentParser()
args_parser.add_argument('--data_path', default='data/', help='path to all train/test/dev folds')
args_parser.add_argument('--max_word_len', type=int, default=150, help='maximum word allowed for 1 instance')
args_parser.add_argument('--max_nb_words', type=int, default=50000, help='maximum size of vocabulary')
args_parser.add_argument('--embedding_dim', type=int, default=300, help='embedding dimension of fasttext')
args_parser.add_argument('--num_class', type=int, default=2, help='number of class')
args_parser.add_argument('--patience', type=int, default=20, help='patience count for early stopping')
args_parser.add_argument('--iterations', type=int, default=100, help='total epoch')
args_parser.add_argument('--batch_size', type=int, default=100, help='total batch size')
args_parser.add_argument('--fasttext_path', default='../cc.id.300.vec', help='path to indonesian fasttext')
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


def model_with_fasttext(x_train, y_train, x_dev, y_dev, x_test, y_test, tokenizer):
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

    # MODEL 
    with tf.device('/gpu:0'):
        embedding_layer = Embedding(nb_words + 1, args.embedding_dim, weights=[embedding_matrix], 
                                input_length=args.max_word_len, trainable=False)
        
        tweet = Input(shape=(args.max_word_len,), dtype='int32')
        embedded_sequences = embedding_layer(tweet)

        lstm_cell = LSTM(units=200, activation='tanh', recurrent_activation='hard_sigmoid', 
                recurrent_regularizer=keras.regularizers.l2(0.2), return_sequences=False, dropout=0.3, recurrent_dropout=0.3)
        doc_vector = Bidirectional(lstm_cell, merge_mode='concat')(embedded_sequences)
        
        sign = Dense(args.num_class, activation='softmax')(doc_vector)
        sent_model = Model([tweet], [sign])
        sent_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

        bestf1=0.0; patience = 0
        for i in range(args.iterations):
            if patience is args.patience:
                break
            sent_model.fit([x_train], [y_train], batch_size=args.batch_size, 
                       epochs=1, shuffle=True, verbose=False)
            prediction=sent_model.predict([x_dev], batch_size=1000)
            predicted_label = np.argmax(prediction,axis=1)
            f1score = f1_score(y_dev,predicted_label)
            if f1score > bestf1:
                print('Epoch ' + str(i) +' with dev f1: '+ str(f1score))
                bestf1 = f1score
                sent_model.save('save.keras')
                patience = 0
            else:
                patience += 1
        sent_model = load_model('save.keras')
        prediction=sent_model.predict([x_test], batch_size=1000)
        predicted_label = np.argmax(prediction,axis=1)
    f1score = f1_score(y_test,predicted_label)
    print('Test F1: ',f1score)
    print('-----------------------------------------------------------------------------------')
    return f1score


def train_and_test_fasttext(x_train, y_train, x_dev, y_dev, x_test, y_test):    
    tokenizer = Tokenizer(num_words=args.max_nb_words, lower=True)
    tokenizer.fit_on_texts(x_train)
    
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)
    x_dev = tokenizer.texts_to_sequences(x_dev)
    
    max_len = max([len(t) for t in x_train])
    print ('Max Len', max_len)
    max_len = args.max_word_len
    x_train = sequence.pad_sequences(x_train, maxlen=max_len, padding='post')
    x_test = sequence.pad_sequences(x_test, maxlen=max_len, padding='post')
    x_dev = sequence.pad_sequences(x_dev, maxlen=max_len, padding='post')
    return model_with_fasttext(x_train, to_categorical(y_train), x_dev, y_dev, x_test, y_test, tokenizer)

print('Batch Size', args.batch_size)
f1s = 0.0
for idx in range(5):
    train = pd.read_csv(args.data_path+'train'+str(idx)+'.csv')
    dev = pd.read_csv(args.data_path+'dev'+str(idx)+'.csv')
    test = pd.read_csv(args.data_path+'test'+str(idx)+'.csv')
    xtrain, ytrain = list(train['sentence']), list(train['sentiment'])
    xdev, ydev = list(dev['sentence']), list(dev['sentiment'])
    xtest, ytest = list(test['sentence']), list(test['sentiment'])
    f1s += train_and_test_fasttext(xtrain, ytrain, xdev, ydev, xtest, ytest)
print('Final Average F1 score in the test set', f1s/5.0)





