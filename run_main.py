#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:03:56 2019

@author: mingliu
"""

#import IE
import timeit
from Graph import *
from utils import get_w2v_embeddings, read_triplets
from transformers import *
import logging
logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)

    
def main():
    start = timeit.default_timer()
    #read all the triplets into a list of lists
    tri_path = "data/multi_news"
    tri_file = "test.src.triplets.txt"
    tri_list = read_triplets(tri_path, tri_file)

    
    #load the LM and word vectors
    #load the LM, tokenizer, and word vectors
    lm_tokenizer = GPT2Tokenizer.from_pretrained('lm_trump')    

    # Models can return full list of hidden-states & attentions weights at each layer
    lm_model = GPT2Model.from_pretrained('lm_trump',
                                  output_hidden_states=True,
                                  output_attentions=False)
    w2v=get_w2v_embeddings('word_vec/news_w2v.txt')  # 278031 word vectors, hidden dim 100
    
    summary_list=[]
    www=0
    for triplets in tri_list[:10]:
        print(www)
        #Build the graph
        tripletGraph = TripletGraph(triplets, lm_model, w2v, lm_tokenizer, use_lm=True, tau=0.5)
        source, target, weight =tripletGraph.build_triplet_graph()

        #Do graph clustering
        clusterIDs = graph_clustering(source, target, weight)
    
        num_clusters = max(clusterIDs)
    
        text_dict = {new_list: [] for new_list in range(num_clusters+1)} 
    
        #loop all the triplets, generate the summary
        for i, clusterID in enumerate(clusterIDs):
            text_dict[clusterID].append(triplets[i])
    
        
        summary=[]
        for k,v in text_dict.items():
            if(len(v)>0):
                summary.append(' '.join(v[0]))
        summary_list.append(' '.join(summary))
        www=www+1
    
    
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    #if(compress):
        
    
    outfile='summary_predict.txt'
    f = open(outfile, "a")
    f.writelines(summary_list)
    f.close()
    print('Done')


if __name__ == "__main__":
    #main()
    '''
    start = timeit.default_timer()
    #read all the triplets into a list of lists
    tri_path = "data/multi_news"
    tri_file = "test.src.triplets.txt"
    tri_list = read_triplets(tri_path, tri_file)

    
    #load the LM, tokenizer, and word vectors
    lm_tokenizer = GPT2Tokenizer.from_pretrained('lm_trump')    

    # Models can return full list of hidden-states & attentions weights at each layer
    lm_model = GPT2Model.from_pretrained('lm_trump',
                                  output_hidden_states=True,
                                  output_attentions=False)
    w2v=get_w2v_embeddings('word_vec/news_w2v.txt')  # 278031 word vectors, hidden dim 100
    
    summary_list=[]
    www=0
    for triplets in tri_list[:10]:
        print(www)
        #Build the graph
        tripletGraph = TripletGraph(triplets, lm_model, w2v, lm_tokenizer, use_lm=True, tau=0.5)
        source, target, weight =tripletGraph.build_triplet_graph()

        #Do graph clustering
        clusterIDs = graph_clustering(source, target, weight)
    
        num_clusters = max(clusterIDs)
    
        text_dict = {new_list: [] for new_list in range(num_clusters+1)} 
    
        #loop all the triplets, generate the summary
        for i, clusterID in enumerate(clusterIDs):
            text_dict[clusterID].append(triplets[i])
    
        print(clusterIDs)
        summary=[]
        for k,v in text_dict.items():
            if(len(v)>0):
                summary.append(' '.join(v[0]))
                print(summary)
        summary_list.append(' '.join(summary))
        www=www+1
    
    
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    #if(compress):
        
    
    outfile='summary_predict.txt'
    f = open(outfile, "a")
    f.writelines(summary_list)
    f.close()
    print('Done')
    '''
    