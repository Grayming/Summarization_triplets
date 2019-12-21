#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:10:34 2019

@author: mingliu
"""

import numpy as np
import os
import spacy
import takahe

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


def tag_pos(str_text):
    doc=spacynlp(str_text)
    textlist=[]
    # compare the words between two strings
    for item in doc:
        source_token = item.text
        source_pos = item.tag_
        textlist.append(source_token+'/'+source_pos)
    return ' '.join(textlist)

def convert_triplet_to_sents(tri_list):
    tagged_list = []
    if(len(tri_list)>0):
        for item in tri_list:
            temp = ' '.join(item)
            temp = temp + '.'
            temp_tagged = tag_pos(temp)
            tagged_list.append(temp_tagged)
    else:
        tagged_list.append(tag_pos('.'))
    return tagged_list
        
def get_compressed_sen(sentences):
    compresser = takahe.word_graph(sentences, 
							    nb_words = 8, 
	                            lang = 'en', 
	                            punct_tag = "." )
    candidates = compresser.get_compression(3)
    reranker = takahe.keyphrase_reranker(sentences,  
									  candidates, 
									  lang = 'en')

    reranked_candidates = reranker.rerank_nbest_compressions()
    #print(reranked_candidates)
    if(len(reranked_candidates)>0):
        score, path = reranked_candidates[0]
        result = ' '.join([u[0] for u in path])
    else:
        result=' '
    return result

   