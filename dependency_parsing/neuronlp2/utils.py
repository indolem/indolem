__author__ = 'max'

from collections import OrderedDict
import pickle, io
import numpy as np
from gensim.models.word2vec import Word2Vec
import gzip, glob, os, json

from neuronlp2.io.common import DIGIT_RE


def load_embedding_dict(embedding, embedding_path, normalize_digits=False, word2index_path=''):
    """
    load word embeddings from file
    :param embedding:
    :param embedding_path:
    :return: embedding dict, embedding dimention, caseless
    """
    print("loading embedding: %s from %s" % (embedding, embedding_path))
    if embedding == 'word2vec':
        # loading word2vec
        word2vec = Word2Vec.load_word2vec_format(embedding_path, binary=True)
        embedd_dim = word2vec.vector_size
        return word2vec, embedd_dim
    elif embedding == 'glove':
        # loading GloVe
        embedd_dim = -1
        embedd_dict = OrderedDict()
        with gzip.open(embedding_path, 'rt') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = line.split()
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                else:
                    assert (embedd_dim + 1 == len(tokens))
                embedd = np.empty([1, embedd_dim], dtype=np.float32)
                embedd[:] = tokens[1:]
                word = DIGIT_RE.sub("0", tokens[0]) if normalize_digits else tokens[0]
                embedd_dict[word] = embedd
        return embedd_dict, embedd_dim
    elif embedding == 'senna':
        # loading Senna
        embedd_dim = -1
        embedd_dict = OrderedDict()
        with gzip.open(embedding_path, 'rt') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = line.split()
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                else:
                    assert (embedd_dim + 1 == len(tokens))
                embedd = np.empty([1, embedd_dim], dtype=np.float32)
                embedd[:] = tokens[1:]
                word = DIGIT_RE.sub("0", tokens[0]) if normalize_digits else tokens[0]
                embedd_dict[word] = embedd
        return embedd_dict, embedd_dim
    elif embedding == 'sskip':
        embedd_dim = -1
        embedd_dict = OrderedDict()
        with gzip.open(embedding_path, 'rt') as file:
            # skip the first line
            file.readline()
            for line in file:
                line = line.strip()
                try:
                    if len(line) == 0:
                        continue

                    tokens = line.split()
                    if len(tokens) < embedd_dim:
                        continue

                    if embedd_dim < 0:
                        embedd_dim = len(tokens) - 1

                    embedd = np.empty([1, embedd_dim], dtype=np.float32)
                    start = len(tokens) - embedd_dim
                    word = ' '.join(tokens[0:start])
                    embedd[:] = tokens[start:]
                    word = DIGIT_RE.sub("0", word) if normalize_digits else word
                    embedd_dict[word] = embedd
                except UnicodeDecodeError:
                    continue
        return embedd_dict, embedd_dim
    elif embedding == 'polyglot':
        words, embeddings = pickle.load(open(embedding_path, 'rb'), encoding='latin1')
        _, embedd_dim = embeddings.shape
        embedd_dict = OrderedDict()
        for i, word in enumerate(words):
            embedd = np.empty([1, embedd_dim], dtype=np.float32)
            embedd[:] = embeddings[i, :]
            word = DIGIT_RE.sub("0", word) if normalize_digits else word
            embedd_dict[word] = embedd
        return embedd_dict, embedd_dim
    elif embedding == 'fasttext':
        fin = io.open(embedding_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        embedd_dict = OrderedDict()
        for line in fin:
            tokens = line.rstrip().split(' ')
            embedd_dict[tokens[0]] = list(map(float, tokens[1:]))
        return embedd_dict, 300
    elif embedding == 'bert':
        assert word2index_path != ''
        with open(word2index_path, 'r') as file:
            word2id = json.load(file)
        embedd_dict = OrderedDict()
        for key in word2id.keys():
            embedd_dict[key] = np.load(embedding_path+'/'+str(word2id[key])+'.npy').tolist()
        print(len(embedd_dict))
        return embedd_dict, 768
    else:
        raise ValueError("embedding should choose from [word2vec, senna, glove, sskip, polyglot]")
