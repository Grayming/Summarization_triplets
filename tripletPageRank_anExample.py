#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11:46:27 2019

@author: mingliu
"""
import IE
import timeit
import os
import numpy as np
from sklearn.cluster import AgglomerativeClustering

    
def get_glove_embeddings(filename):
    word_embeddings = {}
    f = open(filename, encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    return word_embeddings

def get_TripletNode_embedding(triplets,word_embeddings):
    triplet_vectors=[]
    for triplet in triplets:
        sent=str(' '.join(triplet)).lower()
        print(sent)
        for i in sent:
            if len(i) != 0:
                v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
            else:
                v = np.zeros((100,))
        triplet_vectors.append(v)
    return triplet_vectors

def build_similarity_matrix(triplet_vectors):
    from sklearn.metrics.pairwise import cosine_similarity
    sim_mat = np.zeros([len(triplet_vectors), len(triplet_vectors)])
    for i in range(len(triplet_vectors)):
        for j in range(len(triplet_vectors)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(triplet_vectors[i].reshape(1,100), triplet_vectors[j].reshape(1,100))[0,0]
    return sim_mat

def get_triplets_rankings(sim_mat,triplets,n=3):
    import networkx as nx

    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    
    ranked_triplets = sorted(((scores[i],s) for i,s in enumerate(triplets)), reverse=True)
    return ranked_triplets
    '''
    extracted_triplets = []
    for i in range(n):
        extracted_triplets.append(ranked_triplets[i][1])
    return extracted_triplets
    '''
def get_clusterID(triplet_vectors):
    x=np.asarray(triplet_vectors)
    clustering = AgglomerativeClustering().fit_predict(x)
    return clustering
    

extractor = IE.IE_extraction()

s='CT Sinuses performed on 23-DEC-2008 at 01:30 PM:    Technique:  Unenhanced multi detector CT images of the paranasal sinuses with coronal and sagittal reformats.      Findings:  Comparison with similar study dated 17 September 2008.  Minor mucoperiosteal thickening in the anterior and posterior ethmoid air cells bilaterally and the left frontal recess.  The maxillary sinuses, right frontal, and both sphenoid sinuses are clear.  No bony sclerosis or erosion.  The osteomeatal complexes are patent bilaterally.    Impression:  Minor mucoperiosteal thickening in the ethmoid and left frontal sinuses.  No acute or chronic sinusitis.       Co-signed Dr M Pianta. CT Sinuses performed on 30-MAR-2009 at 09:28 AM:    CT scan of the para nasal sinuses.    History:    Exclude fungal disease.  Chemotherapy for ALL.    Procedure:    Helical scans were performed through the para nasal sinuses with sagittal and coronal reformat images.    Findings:    There is a trace of mucosal thickening seen in the dependent portion of both maxillary antra.  The osteomeatal complex is patent bilaterally.  The remaining para nasal sinuses are clear.  No evidence of erosion of the sinus walls.  The nasal cavity is unremarkable.    Comment:    No significant sinus pathology identified.  Specifically no evidence of fungal sinusitis.'

start = timeit.default_timer()
triplets = extractor.final_triplets(s)
stop = timeit.default_timer()
print('Time: ', stop - start)


word_embeddings = get_glove_embeddings('fungal_w2v.txt')

triplet_vectors = get_TripletNode_embedding(triplets, word_embeddings)
#print(triplet_vectors[0])
clusterID=get_clusterID(triplet_vectors)
for i in range(0,len(triplets)):
    print(clusterID[i])
    print(triplets[i])
print(clusterID)
sim_mat = build_similarity_matrix(triplet_vectors)
print(sim_mat)
ranked_triplets = get_triplets_rankings(sim_mat,triplets,n=3)
print(ranked_triplets)








