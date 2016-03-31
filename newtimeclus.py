#! /usr/bin/env python

#A time based clustering implementation for document clustering using TF_IDF.
#Script for one folder.

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
import numpy as np
import copy
import cPickle as pickle
import json
from datetime import datetime
import time

#create folder for current time
t = datetime.now()
foldername = t.strftime('%m_%d_%Y_%H_%M_TimeBased_Extracts')
tbepath = "experimental_tbe/"+foldername
if not os.path.exists(tbepath):
	os.mkdir("experimental_tbe/"+foldername)

worddict = None

#See if we have a word lookup vector
try:
        worddict = pickle.load(open("worddict.dict","rb"))
except:
	#Throw Error to Log
	print "Error: worddict not found"
	exit()
        worddict = [];

#build idf from the whole matrix
def idf_build(wmatrix):
	#make a boolean matrix to simulate the IDF calculation process:
	#If an entry's TF is nonzero, then it must have occurrence of that word
	#So we use the boolean sum along the column to see how many articles have that word 
	idf = (wmatrix>0.0).astype(float).sum(0)
	#df is simply the number of columns
	df,r = wmatrix.shape
	#+0.00000001 bias to avoid Division by Zero
	idf = float(df)/(idf+0.001)
	idf = np.log(idf)
	return idf

ind = 0
fmatrix = None
tbelist = None
fname_lookup = [];

#See if we have a 40-batch cluster ready
try:
	(ind,lngth,tbelist) = pickle.load(open("tbelist.lst","rb"))
	if lngth < 40:
		#Log: Wait for 40 batches
		exit()
except:
	#Throw error to log
	exit()

#standardize vector length
vlength = len(worddict)

#print tbelist
#exit()
first = True
for mpath in tbelist:
	#read matrix
	mat = pickle.load(open("experimental/"+mpath+"/matrix.npy",'r'))
	c,r = mat.shape
	if first:
		#reshape the matrix to length of wordvector
		print mpath
		fmatrix = np.column_stack([mat,[[0.0]*(vlength-r)]*c])
	else:
		#reshape
		tmatrix = np.column_stack([mat,[[0.0]*(vlength-r)]*c])
		#stack below the original matrix
		print mpath
		fmatrix.shape
		tmatrix.shape
		fmatrix = np.vstack([fmatrix,tmatrix])
	first = False
	#update the lookup
	#read from dump
	fnlookup = pickle.load(open("experimental/"+mpath+"/fname_lookup.txt"))
	#update the list to include the datapath
#	fnlookup = map(lambda x: mpath+"/"+x,fnlookup);
	#append to the lookup;
	fname_lookup += fnlookup;

#multiply tf and idf
tfidf = np.multiply(fmatrix,idf_build(fmatrix))
	
#pdist cosine to extract closest news
distmatrix = np.around(scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(tfidf,"cosine")),decimals=12)

sort_batch = batch_dist.reshape(-1).argsort()
columns,dontcare = batch_dist.shape

#For each return closest 7
#recommendations dict
rec_vec = {}

df,r = fmatrix.shape 

for i in range(0, df):
        aind = 0
        vec = []
        a = distmatrix[i]
        #sort and get 3 closest: one is self.
        l = np.argpartition(a, range(0,df))
        for lind in range(0,df):
                #not self; not identical
                if a[l][lind] > 0.00001:
                        vec += [fname_lookup[l[lind]]]
                        aind += 1
                        if (aind == 7):
                                break
        rec_vec[fname_lookup[i]] = vec

#JSON Dump
#log the dump list
rec_vec["clustered_list"] = tbelist
#dump rec_vec
with open(tbepath+'/recommendationlist.json', 'w') as fp:
        json.dump(rec_vec, fp)
#dump matrix
fmatrix.dump(tbepath+'/stackedmatrix.npy')
distmatrix.dump(tbepath+'/distancematrix.npy')
#dump filelist
with open(tbepath+'/tbelist.txt', 'w') as fp:
        for f in fname_lookup:
		fp.write(f+"\n");

#Optional: dump the scores(dumping pairwise consumes too much space.)
menuframe = os.path.join(tbepath,'menu.html')
menu = open(menuframe,'w')

menu.write('<html>\n<head>\n<title>Extraction</title>\n<style type="text/css">\nbody {\nfont-family:verdana,arial,sans-serif;\nfont-size:10pt;\nmargin:10px;\nbackground-color:#ff9900;\n}\n</style>\n</head>\n<body>\n<h3>Menu 1</h3>\n')

#for each pair of reports
for t in sort_batch:
        x = t/columns
        y = t%columns
        if x>y:
		xlink = '<a href="../../Data/'+fname_lookup[x]+'">'+ fname_lookup[x]+"</a>";
		if fname_lookup[x].split('/')[0] == fname_lookup[y].split('/')[0]:
			continue 
		ylink = '<a href="../../Data/'+fname_lookup[y]+'">'+ fname_lookup[y]+"</a>";
		menu.write("<p>"+distmatrix[x][y].astype('|S10')+"   "+xlink+"   "+ylink+"</p>");
menu.write('</html>')
menu.close();


