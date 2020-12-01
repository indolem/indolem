#!/usr/bin/env python
# coding: utf-8

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score,accuracy_score
from sklearn.exceptions import ConvergenceWarning
import pandas as pd
import numpy as np
from bpe import Encoder
import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# Naive Bayes
alpha_vals = np.linspace(0.001,1,100)
def tuning_nb(xtrain, ytrain, xdev, ydev):
    alpha_params = {}
    for alpha in alpha_vals:
        model = MultinomialNB(alpha = alpha)
        model.fit(xtrain,ytrain)
        pred = model.predict(xdev)
        alpha_params[alpha] = f1_score(ydev,pred)
    maxAlpha = max(alpha_params, key=alpha_params.get) 
    print('After tuning NB for', len(alpha_vals), 'we get param:', maxAlpha)
    return maxAlpha

def train_and_test_nb(xtrain, ytrain, xtest, ytest, xdev, ydev):
    encoder = Encoder(5000, pct_bpe=0.88)
    encoder.fit(xtrain)
    xtrain = [' '.join(encoder.tokenize(name)) for name in xtrain]
    xtest = [' '.join(encoder.tokenize(name)) for name in xtest]
    xdev = [' '.join(encoder.tokenize(name)) for name in xdev]
    
    vectorizer = CountVectorizer(ngram_range=(1, 2), lowercase=False)
    x_train = vectorizer.fit_transform(xtrain)
    x_test = vectorizer.transform(xtest)
    x_dev = vectorizer.transform(xdev)
    
    maxAlpha = tuning_nb(x_train, ytrain, x_dev, ydev)
    clf = MultinomialNB(maxAlpha)
    clf.fit(x_train.toarray(), ytrain)
    pred = clf.predict(x_test.toarray())
    f1score = f1_score(ytest,pred)
    return f1score


# Logistic Regression
c_vals = np.linspace(0.001,10,100)
def tuning_lr(xtrain, ytrain, xdev, ydev):
    c_params = {}
    for c in c_vals:
        model = LogisticRegression(C = c)
        model.fit(xtrain,ytrain)
        pred = model.predict(xdev)
        c_params[c] = f1_score(ydev,pred)
    maxC = max(c_params, key=c_params.get) 
    print('After tuning LR for', len(alpha_vals), 'we get param:', maxC)
    return maxC

def train_and_test_lr(xtrain, ytrain, xtest, ytest, xdev, ydev):
    encoder = Encoder(1000, pct_bpe=0.88)
    encoder.fit(xtrain)
    xtrain = [' '.join(encoder.tokenize(name)) for name in xtrain]
    xtest = [' '.join(encoder.tokenize(name)) for name in xtest]
    xdev = [' '.join(encoder.tokenize(name)) for name in xdev]
    
    vectorizer = CountVectorizer(ngram_range=(1, 2), lowercase=False)
    x_train = vectorizer.fit_transform(xtrain)
    x_test = vectorizer.transform(xtest)
    x_dev = vectorizer.transform(xdev)
    
    maxC = tuning_lr(x_train, ytrain, x_dev, ydev)
    clf = LogisticRegression(C=maxC)
    clf.fit(x_train.toarray(), ytrain)
    pred = clf.predict(x_test.toarray())
    f1score = f1_score(ytest,pred)
    return f1score


args_parser = argparse.ArgumentParser()
args_parser.add_argument('--data_path', default='data/', help='path to all train/test/dev folds')
args = args_parser.parse_args()

print('Naive Bayes with 5-fold Cross Validation')
f1s = 0.0
for idx in range(5):
    train = pd.read_csv(args.data_path+'train'+str(idx)+'.csv')
    test = pd.read_csv(args.data_path+'test'+str(idx)+'.csv')
    dev = pd.read_csv(args.data_path+'dev'+str(idx)+'.csv')
    xtrain, ytrain = list(train['sentence']), list(train['sentiment'])
    xtest, ytest = list(test['sentence']), list(test['sentiment'])
    xdev, ydev = list(dev['sentence']), list(dev['sentiment'])
    f1s += train_and_test_nb(xtrain, ytrain, xtest, ytest, xdev, ydev)
print('Final Average F1 score in the test set', f1s/5.0)
print()

print('Logistic Regression with 5-fold Cross Validation')
f1s = 0.0
for idx in range(5):
    train = pd.read_csv(args.data_path+'train'+str(idx)+'.csv')
    test = pd.read_csv(args.data_path+'test'+str(idx)+'.csv')
    dev = pd.read_csv(args.data_path+'dev'+str(idx)+'.csv')
    xtrain, ytrain = list(train['sentence']), list(train['sentiment'])
    xtest, ytest = list(test['sentence']), list(test['sentiment'])
    xdev, ydev = list(dev['sentence']), list(dev['sentiment'])
    f1s += train_and_test_lr(xtrain, ytrain, xtest, ytest, xdev, ydev)
print('Final Average F1 score in the test set', f1s/5.0)

