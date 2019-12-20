#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:10:34 2019

@author: mingliu
"""

import numpy as np
import os
import spacy

spacynlp = spacy.load("en_core_web_sm")

def get_w2v_embeddings(filename):
    word_embeddings = {}
    f = open(filename, encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    return word_embeddings


# read triplets into a list of list
def read_triplets(path, file_name):
    f = open(os.path.join(path, file_name),"r")
    lines = f.readlines()
    tri_list = []
    for line in lines:
        data = line.split("\t")
        temp = [ d.split("|") for d in data]
        tri_list.append(temp)
    return tri_list


def slash_pos(str_text):
    doc=spacynlp(str_text)
    textlist=[]
    # compare the words between two strings
    for item in doc1:
        source_token = item.text
        source_pos = item.pos_
        textlist.append(source_token+'/'+source_pos)
    return ' '.join(textlist)
    
    