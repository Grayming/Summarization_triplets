#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 14:02:26 2019
The LexRank algorithm, used for extractive text summarization.

@author: mingliu
"""
import timeit
from lexrank import STOPWORDS, LexRank
from path import Path
import spacy
from spacy.lang.en import English

nlp = English()
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)
nlp.max_length = 2000000

start = timeit.default_timer()
documents = []
documents_dir = Path('data/multi_news')

for file_path in documents_dir.files('*.src'):
    with file_path.open(mode='rt', encoding='utf-8') as fp:
        documents.append(fp.readlines())

lxr = LexRank(documents, stopwords=STOPWORDS['en'])

source_path = 'data/multi_news/test.txt.src'
source_files=open(source_path, 'r').readlines()

all_source_files = []
all_summary = []
for item in source_files:
    doc = nlp(item, disable = ['ner', 'parser'])
    sents=[]
    for sent in doc.sents:
        sents.append(sent.text)
    summary = lxr.get_summary(sents, summary_size=11, threshold=0.1)
    all_summary.append(summary)
print(len(all_summary))
outfile='output/summary_predict_lexrank.txt'
with open(outfile, 'w') as f:
    for item in all_summary:
        f.write("%s\n" % item)
print('Done')


stop = timeit.default_timer()
print('Time: ', stop - start)













