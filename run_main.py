#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:03:56 2019

@author: mingliu
"""

#import IE
from Graph import *
from utils import get_w2v_embeddings, read_triplets

def demo():
    #read all the triplets into a list of lists
    tri_path = "data/multi_news"
    tri_file = "test.src.triplets.txt"
    tri_list = read_triplets(tri_path, tri_file)

    
    #load the LM and word vectors
    lm_model=''
    w2v=get_w2v_embeddings('word_vec/news_w2v.txt')  # 278031 word vectors, hidden dim 100
    
    summary_list=[]
    #for triplets in tri_list:
    triplets=tri_list[0]
    #Build the graph
    tripletGraph = TripletGraph(triplets, lm_model, w2v, tau=0.5)
    source, target, weight =tripletGraph.build_triplet_graph()

        #Do graph clustering
    clusterIDs = graph_clustering(source, target, weight)
    
    num_clusters = max(clusterIDs)
    
    text_dict = {new_list: [] for new_list in range(num_clusters+1)} 
    
        #loop all the triplets, generate the summary
    for i, clusterID in enumerate(clusterIDs):
        text_dict[clusterID].append(triplets[i])
    
    print(text_dict)
    
    summary=[]
    for k,v in text_dict.items():
        if(len(v)>0):
            summary.append(' '.join(v[0]))
    print(' '.join(summary))
    
def main():
    #read all the triplets into a list of lists
    tri_path = "data/multi_news"
    tri_file = "test.src.triplets.txt"
    tri_list = read_triplets(tri_path, tri_file)

    
    #load the LM and word vectors
    lm_model=''
    w2v=get_w2v_embeddings('word_vec/news_w2v.txt')  # 278031 word vectors, hidden dim 100
    
    summary_list=[]
    for triplets in tri_list:
        #Build the graph
        tripletGraph = TripletGraph(triplets, lm_model, w2v, tau=0.5)
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
    
    outfile='summary_predict.txt'
    f = open(outfile, "a")
    f.writelines(summary_list)
    f.close()
    print('Done')


if __name__ == "__main__":
    main()
    
    