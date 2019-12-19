#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 20:21:21 2019

@author: mingliu
"""
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from matplotlib import pyplot
import pandas as pd
from nltk.tokenize import RegexpTokenizer
import sys
#reload(sys)
#sys.setdefaultencoding("ISO-8859-1") 


tokenizer = RegexpTokenizer(r"\w+(?:[-.]\w+)?")

def getSentences(filename):
    docs=open(filename, encoding='utf-8', errors='replace').readlines()
    sentences=[]
    for doc in docs:
        sentence=doc.rstrip().strip().lower()
        tokens=tokenizer.tokenize(sentence)
        if(len(tokens)>10):
            sentences.append(tokens)
    return sentences

def getw2v(sentences):
    #path = get_tmpfile("word2vec.model")
    
    model = Word2Vec(sentences, size=100, window=2, min_count=2, workers=4)
    model.save("word_vec/news_word2vec.model")
    model.wv.save_word2vec_format('word_vec/news_w2v.txt', binary=False)
    '''
    # fit a 2d PCA model to the vectors
    X = model[model.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    # create a scatter plot of the projection
    pyplot.scatter(result[:, 0], result[:, 1])
    words = list(model.wv.vocab)[:50]
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()
    '''

if __name__== "__main__":
  sentences=getSentences('data/multi-news/train.txt.src')
  getw2v(sentences)
  #print(sentences)