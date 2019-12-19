#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:03:56 2019

@author: mingliu
"""

#import IE
from Graph import *
from utils import get_glove_embeddings, read_triplets
import timeit
import numpy as np

def main():
    #read all the triplets into a list of lists
    tri_path = "data/multi_news"
    tri_file = "test.src.triplets.txt"
    tri_list = read_triplets(tri_path, tri_file)

    print(tri_list[0])
    '''
    #load the LM and word vectors
    lm_model=''
    w2v=get_glove_embeddings('word_vec/news_w2v.txt')  # 278031 word vectors, hidden dim 100

    #Build the graph
    tripletGraph = TripletGraph(tri_list, lm_model, w2v, tau=0.5)
    source, target, weight =tripletGraph.build_triplet_graph()

    #Do graph clustering
    clusterID = graph_clustering(source, target, weight)
    
    
    '''


if __name__ == "__main__":
    main()
