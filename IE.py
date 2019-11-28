#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11:44:08 2019

@author: qipan
"""

from pycorenlp import StanfordCoreNLP
from nltk import sent_tokenize
from pyclausie import ClausIE
import string



class IE_extraction:
    def __init__(self,):
        self.stanford_model = StanfordCoreNLP("http://localhost:9000/")
        self.stanford_all_annotate("")
        self.content = None

    def stanford_all_annotate(self,text):
        
        self.content = self.stanford_model.annotate(text,
                        properties={
                            'annotators': 'ssplit, pos, depparse, openie, dcoref',
                            'outputFormat': 'json',
                            'timeout': 50000,
                        })  

    # extract coref   
    # original coreference chain
    def stanford_coref(self,):
        corefs = []
        
        for coref in self.content['corefs'].values():
            # test each coref sublist contains more than 1 element
            if len(coref) > 1:
                f = []
                for j in coref:
                    f.append([j['text'], j['type'], j['sentNum'], j['startIndex']])
                corefs.append(f)
        return corefs

    # tokenise
    def stanford_token(self,):
        tokens = []
        for sentence in self.content['sentences']:
            re = [t['word'] for t in sentence['tokens']]
            tokens.append(re)
        return tokens 


    #extract triple by openie
    def openie_extraction(self,k):
        triple = []
        for sentence in self.content['sentences']:
            result = sentence["openie"] 
            for relation in result:
                triple.append([k,relation['subject'], relation['relation'], relation['object']])
        return triple

    
    # def nltk_sentence_token(self,text):
    #     self.sentences = sent_tokenize(text)
        
    # extract by clausie
    # warning do coref and token by stanfordnlp
    def clausie(self,k,text):
        cl = ClausIE.get_instance()
        sentences = sent_tokenize(text)
        triplets = []
        for sent in sentences:
            try:
                extraction = cl.extract_triples([sent])
                for i in extraction:
                    triplets.append([str(k)] + list(i[1:4]))
            except ValueError:
                pass
        
        return triplets

    def get_index(self,triplets):
        return triplets[0]

    # split sentences by stanford nlp
    def stanford_sentence_split(self,):
        tokens = self.stanford_token()
        return ["".join([" " + i if not i.startswith("'") and i not in string.punctuation else i for i in token]).strip() for token in tokens]

    def coref_resolve(self,):
        pronouns = {'i', 'he', 'she', 'it', 'they', 'this', 'my', 'its', "it's", 'me', 'him', 'his', 'that', 'their', 'her', 'which', 'those','himself','herself','itself','we','us','our','ourselves','them','their','themselves'}
        corefs = self.stanford_coref()

        re = []
        # delete coreference whose recourse is in pronouns
        # delete coreference whose destination is not in pronouns
        for coref in corefs :
            if not coref[0][0].lower() in pronouns:
                holder = [coref[0]]
                for i in coref[1:]:
                    if i[0].lower() in pronouns:
                        holder.append(i)
                if len(holder) > 1:
                    re.append(holder)
        return re


    #untokenize a sentence tokens
    def untokenize(self,toks):
        import string
        untok = "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in toks]).strip()
        return untok


    def coref_sentence(self,):
        sentences = self.stanford_token()
        corefs = self.coref_resolve()
        for coref in corefs:
            source = coref[0][0]
            for k in coref[1:]:
                sentences[k[2]-1][k[3]-1] = source
        
        return [self.untokenize(i) for i in sentences]
    

    
 
    def final_triplets(self,text):

        def reduce_triplet(temptriplets):
            i=0
            j=0
            for k in range(0,len(temptriplets)):
                triplet_string=' '.join(temptriplets[k][1:])
                
                if(len(triplet_string)>i):
                    i=len(triplet_string)
                    j=k
            return temptriplets[j][1:]
            #return temptriplets[j][1:]
        
        # extract sentences after coreference
        self.stanford_all_annotate(text)
        sentences = self.coref_sentence()
        result = []
        # extract triples of each sentences
        for k,sent in enumerate(sentences):
            self.stanford_all_annotate(sent)
            temptriplets= self.openie_extraction(k) + self.clausie(k,sent)
            if(len(temptriplets)>0):
                result.append(reduce_triplet(temptriplets))
            #result += self.openie_extraction(k)
            #result += self.clausie(k,sent)
        return result


