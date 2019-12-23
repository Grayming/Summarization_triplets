#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 17:23:00 2019
Given a list of triplet lists, this script is used for triplet 
graph construction and clustering. 

Use class TripletGraph to do graph construction.
 
use graph_clustering (Graclus) for graph clustering.
@author: mingliu
"""

#import networkx as nx
import numpy as np
import scipy
from scipy import *
import torch
from torch_cluster import graclus_cluster
import spacy

spacynlp = spacy.load("en_core_web_sm")
import gensim.downloader as api

glove_word_vectors = api.load("glove-wiki-gigaword-100")

def compare_word(word_vectors, w1, w2):
    flag = False
    try:
        nn1 = word_vectors.most_similar(positive = [w1])
        nn2 = word_vectors.most_similar(positive = [w2])
        #print(nn1)
        #print(nn2)
        for i in nn1:
            word1 = i[0]
            for j in nn2:
                word2 = j[0]
                if (word1 in word2 or word2 in word1):
                    flag = True
                else:
                    continue
    except KeyError:
        pass
    
    return flag
        

# compute the cos similarity between a and b. a, b are numpy arrays
def cos_sim(a, b):
    #return np.dot(a,b) / ((np.dot(a,a) **.5) * (np.dot(b,b) ** .5))
    return 1 - scipy.spatial.distance.cosine(a,b)

def compare_string(word_vectors, str1, str2):
    flag=False
    doc1=spacynlp(str1)
    doc2=spacynlp(str2)
    # compare the words between two strings
    for i in doc1:
        source_token = i.text
        source_pos = i.pos_
        
        for j in doc2:
            target_token = j.text
            target_pos = j.pos_
            if(source_token in target_token or target_token in source_token):
                flag=True
            if((source_pos is 'NOUN' and target_pos is 'NOUN') or (source_pos is 'PROPN' and target_pos is 'PROPN')):
                flag = compare_word(word_vectors, source_token, target_token)
    return flag

#
def graph_clustering(source, target, weight):
    row = torch.tensor(source)
    col = torch.tensor(target)
    weight = torch.Tensor(weight)  # Optional edge weights.

    cluster = graclus_cluster(row, col, weight)
    return cluster.tolist()
    
    
    
    
    
    
class TripletGraph:
    
    def __init__(self, triplet_list, lm_model, w2v, tokenizer, use_lm = True, tau =0.5):
        
        self.triplets = list(triplet_list)
        """ A list of triplets provided by the user. """

        self.length = len(triplet_list)
        """ The number of triplets given for clustering. """
        
        self.lm_model = lm_model
        
        self.w2v = w2v
        
        self.tokenizer = tokenizer
        
        self.use_lm = use_lm
        
        self.tau = tau
        
    def get_wv_embedding(self, string):
        word_embeddings = self.w2v
        sent=string.lower()
        #print(sent)
        if len(sent) != 0:
            vectors = [word_embeddings.get(w, np.zeros((100,))) for w in sent.split()]
            v=np.mean(vectors, axis=0)
        else:
            v = np.zeros((100,))
        return v
    
    def get_lm_embedding(self, string):
        
        sent=string.lower()
        if len(sent)!= 0:
            input_ids = torch.tensor([self.tokenizer.encode(sent)])
            #last_hidden_state, presents,all_hidden_states, all_attentions = model(input_ids)
            last_hidden_state = self.lm_model(input_ids)[0]

            hidden_state=last_hidden_state.tolist()
            v = np.mean(hidden_state,axis=1)
        else:
            v = np.zeros((768,))
        return v
        
    def build_triplet_graph(self):
        '''
        Constructs a directed triplet graph from the list of input triplets. Each
        triplet is iteratively added to the graph according to the 
        following algorithm:
        '''
        source=[]
        target=[]
        weight=[]
        for i in range(self.length):
            s1_head = self.triplets[i][0]
            if self.use_lm:
                s1_head_emb = self.get_lm_embedding(s1_head)
                s1_tail_emb = self.get_lm_embedding(self.triplets[i][2])
            else:
                s1_head_emb = self.get_wv_embedding(s1_head)
                s1_tail_emb = self.get_wv_embedding(self.triplets[i][2])
            
            for j in range(self.length):
                s2_head = self.triplets[j][0]
                if self.use_lm:
                    s2_head_emb = self.get_lm_embedding(s2_head)
                    s2_tail_emb = self.get_lm_embedding(self.triplets[j][2])
                else:
                    s2_head_emb = self.get_wv_embedding(s2_head)
                    s2_tail_emb = self.get_wv_embedding(self.triplets[j][2])
                
                flag=False
                # compare head embedding similarity
                if(cos_sim(s1_head_emb, s2_head_emb) > self.tau):
                    flag=True
                # ADG
                if(len(s1_head)>0 and len(s2_head)>0):
                    flag=compare_string(glove_word_vectors, s1_head, s2_head)
                # also add condition: if j-i=1, there are linkage words such as "how"
                #if (j-i is 1):
                if(flag and i is not j):
                    source.append(i)
                    target.append(j)
                    # for the weight, chage it into the normalized version
                    weight.append(cos_sim(s1_tail_emb, s2_tail_emb))
                   
                else:
                    continue
            
        return source, target, weight
    '''
    def build_sentence_graph(self):
        G = nx.Graph()
        for i in range(self.length):
            s1 = ' '.join(self.triplets[i])
            s1_emb = self.get_wv_embedding(s1)
            for j in range(self.length):
                s2 = ' '.join(self.triplets[j])
                s2_emb = self.get_wv_embedding(s2)
                
                if(cos_sim(s1_emb, s2_emb) > self.tau and i is not j):
                    G.add_edge(i, j,weight=cos_sim(s1_emb, s2_emb))
                else:
                    continue
        
        return G
     '''   
    
   
    

    
    
        