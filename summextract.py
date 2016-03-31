#!/usr/bin/env python

import networkx as nx
import nltk
import os
import re
import sys
import math
from collections import defaultdict
import subprocess
import operator
import scipy.spatial
import scipy.cluster
import copy 
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import numpy as np

#build tfisf dictionary
def idf_build(tokenlist,df):
        idf = defaultdict(lambda:0)
        for t in tokenlist:
                idf[t] += 1
        worddict = [x for x,y in idf.items()]
        idf  = np.array([float(df)/y for x,y in idf.items()])
        #worddict is the lookup;
        #idf is the numpy idf vector
        idf = np.log(idf)
        return (idf,worddict);

def tf_build(sentset,idf,lookup):
        mat = range(0,len(sentset))
        for r,ps,ind in sentset:
                vec = [0.0]*len(lookup)
                slen = len(ps)
                for t in ps:
                        vec[lookup.index(t)] += 1
                mat[ind] = vec
                if slen > 0.000001:
                        mat[ind] = [float(x)/slen for x in mat[ind]]
        tf = np.multiply(np.array(mat),idf)
        return tf

#load stopwords list
stopwords = nltk.corpus.stopwords.words('english')

#load up the sentence parser
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

#Define Regex cleaner for step 3.
regex = re.compile('[:-`\\@\=(),\.!\?\;"\']')

#Define Stemmer
stemmer = nltk.PorterStemmer()

#input file path in arguments
path = os.sys.argv[1]

datapath = "Data/"+path
outputpath = "experimental/"+path+"/summary/"

pathlist = [fname for fname in os.listdir(datapath) if fname.endswith(".txt")]

for fp in pathlist:
    doc = open(datapath+"/"+fp,'r')
    outfile = open(outputpath+"/"+fp,'w')

    #purge title line
    title = ['\n']
    while (title == ['\n']):
            title = doc.readline().replace('\n','')
    #sentence
    sentlist = tokenizer.tokenize(doc.read());

    for token in sentlist:
    	if "reporting by" in token.lower():
    		sentlist.remove(token)

    #deep copy
    final_sentlist = copy.deepcopy(sentlist)
    doc.close()

    if not sentlist:
    	print>>outfile, "This is an empty document from Yahoo!"
	outfile.close()
    	continue

    #Step 1: separate sentences
    sentlist = [y.strip() for x in sentlist for y in tokenizer.tokenize(x.replace('\n','')) if y != '']

    #proc_sentlist = copy.deepcopy(sentlist)

    #Step 2: Case Folding
    sentlist = [t.lower() for t in sentlist]

    #processed outfile.
    """if prep_dbg:
            processed = open("prep_"+path,'w')"""

    #list of sentences and preprocessed form
    los=[None]*len(sentlist);
    index = 0
    #Steps 3 and 4
    for sent in sentlist:
            nlist = sent.split()
            #remove all special characters in the sentence.
            nlist = [regex.sub('', token) for token in nlist]
            #Step 3: Remove stopwords
            nlist = [t for t in nlist if t not in stopwords]
            #Step 4: Stemming (I.E. Removing suffixes)
            nlist = [stemmer.stem(t) for t in nlist]
            nlist = [t for t in nlist if t != '']
            #Step 5: Print/Write the result.
            los[index] = (sent,nlist,index);
            #Dump to preprocessed dump
            """if prep_dbg:
    	        for t in nlist:
                            processed.write(t)
                            processed.write(" ")
                    #If all words are removed
                    if not nlist:
              	        processed.write("notokenshere")
                   		processed.write("\n")"""
            index += 1

    #rest of preprocessed data
    prep = []
    for x,y,z in los:
    	prep += list(set(y))

    #whole preprocessed document for tfisf lookup construction
    prep = [y for y in prep if y != '']
    if len(los) == 0:
            print fname,"no sentence"
	    outfile.close()
            continue
    if len(prep) == 0:
            print fname,"no token"
            outfile.close()
    	    continue
    isf,tfisf_dict = idf_build(prep,len(los))

    #Consruct the tf part;
    #gets the matrix out
    tf = tf_build(los,isf,tfisf_dict);

    tf_graph = nx.from_numpy_matrix(np.dot(tf,tf.transpose()))
    scores = nx.pagerank(tf_graph)
    rankings = sorted(((scores[i],s) for i,s in enumerate(final_sentlist)), reverse=True)

    summlen = int((len(los)*0.25)+1.0)

    summarylist = rankings[0:summlen]
    summary = [y for x,y in summarylist]

    print>>outfile, title

    for token in final_sentlist:
    	if token in summary:
    		print>>outfile, token
    outfile.close()
