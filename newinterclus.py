#! /usr/bin/env python

#A new implementation for document clustering using TF_IDF.
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

#General Flow:
#1. remove stopwords
#2. clean punctuations with Regex filter
#3. stemming
#4. build TFIDF for one single document
#5. build matrix
#6. UPGMA clustering

#build idf dictionary
def idf_build(tokenlist,df,worddict,idf):
        for t in set(tokenlist):
		if (t!="thisisempty"):
                	idf[t] += 1
        worddict =worddict + [x for x,y in idf.items() if x not in worddict]
	idf_tpl = [idf[t] for t in worddict]
        idf_vector = np.array([float(df)/(0.001+y) for y in idf_tpl])
        #worddict is the lookup;
        #idf is the numpy idf vector
        idf_vector = np.log(idf_vector)
        return (idf,idf_vector,worddict);
def tf_build(sentset,lookup):
        mat = range(0,len(sentset))
        for ps,ind in sentset:
		vec = [0.0]*len(lookup)
                slen = len(ps)
                for t in ps:
			if (t!= "thisisempty"):
                        	vec[lookup.index(t)] += 1
		mat[ind] = vec
                mat[ind] = [float(x)/slen for x in mat[ind]]
        tf = np.array(mat)
        return tf

#load stopwords list
stopwords = nltk.corpus.stopwords.words('english')

#load up the sentence parser
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

#Define Regex cleaner.
regex = re.compile('[\*\n\r\=(),\.!\?\;"\']')

#Define Stemmer
stemmer = nltk.PorterStemmer()

#input file path in arguments
path = os.sys.argv[1]
datapath = "Data/"+path

worddict = None

#See if we have a word lookup vector
try:
	worddict = pickle.load(open("worddict.dict","rb"))
except:
	worddict = [];

#Process files
pathlist = [fname for fname in os.listdir(datapath) if fname.endswith(".txt")]
ind = 0;
doc_freq = len(pathlist)
fname_lookup = [None]*len(pathlist)
ftext_lookup = [None]*len(pathlist)
idf = defaultdict(lambda:0)
idf_vec = None

#Generate IDF vector and update word vector of the whole database.
for fname in pathlist:
	doc = open(datapath+"/"+fname,'r')
	#purge title line
	title = ['\n']
	while (title == ['\n']):
        	title = doc.readline()
	sentlist = tokenizer.tokenize(doc.read());
	if not sentlist:
		sentlist = ["thisisempty"]
	sentlist = [t.lower() for t in sentlist]
	sentlist = [sent.split() for sent in sentlist]
	sentlist = [item for sublist in sentlist for item in sublist]

	sentlist = [regex.sub('', token) for token in sentlist]
#	sentlist = [t for t in sentlist if t not in stopwords]
	sentlist = [stemmer.stem(t) for t in sentlist]

	idf,idf_vec,worddict = idf_build(sentlist,doc_freq,worddict,idf);
	fname_lookup[ind] = fname;
	ftext_lookup[ind] = (sentlist,ind);
	ind += 1;


#Build the TF matrix 
tf = tf_build(ftext_lookup,worddict)	

#save it for time based clustering
tf.dump('experimental/'+path+"/matrix.npy")
tf = np.multiply(tf,idf_vec)
#batch pairwise distance
batch_dist = np.around(scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(tf,'cosine')),decimals=12)

sort_batch = batch_dist.reshape(-1).argsort()
columns,dontcare = batch_dist.shape

#recommendations dict;
rec_vec = {}

for i in range(0,ind):
	aind = 0
	vec = []
	a = batch_dist[i]
	#sort and get 3 closest: one is self.
	l = np.argpartition(a, range(0,ind))
	for lind in range(0,ind):
		#not self; not identical
		if a[l][lind] > 0.00001:
			vec += [fname_lookup[l[lind]]]
			aind += 1
			if (aind == 4):
				break
	rec_vec[fname_lookup[i]] = vec
#print rec_vec
with open('experimental/'+path+'/recommendationlist.json', 'w') as fp:
	json.dump(rec_vec, fp)

#dump fnamelookup
fnamel = [path+'/'+fn for fn in fname_lookup]
pickle.dump(fnamel,open("experimental/"+path+"/fname_lookup.txt","wb"));

#dump the updated worddict out
pickle.dump(worddict,open("worddict.dict","wb"))

tbtuple = None
#Update time base index
try:
	tbtuple = pickle.load(open("tbelist.lst","rb"))
except:
	tbtuple = (0,0,[[]]*40)
x,z,y = tbtuple
x = x%40;
y[x] = path;
x += 1;
if z < 40:
	z += 1;
tbtuple = (x,z,y)
pickle.dump(tbtuple,open("tbelist.lst","wb"))

#Generate HTML debug pages
#HTML Base datapath
htmlbase = "experimental/"+path+"/html"

#Make the htmlbase folder
if not os.path.exists(htmlbase):
    os.makedirs(htmlbase);

#main frame
mainframe = open(os.path.join(htmlbase,'mainframe.html'),'w');

mainframe.write('<html><head><title>mainframe</title></head>\n<frameset cols="20%,*" frameborder="0" border="0" framespacing="0">\n     <frame name="menu" src="menu.html" marginheight="0" marginwidth="0" scrolling="auto" noresize>\n<frame name="content" src="../../../../index.html" marginheight="0" marginwidth="0" scrolling="auto" noresize>\n<noframes>\n<p>Nope</p>\n</noframes>\n</frameset>\n</html>\n')

mainframe.close()

#menu(leftbar)
menuframe = os.path.join(htmlbase,'menu.html')
menu = open(menuframe,'w')

menu.write('<html>\n<head>\n<title>Extraction</title>\n<style type="text/css">\nbody {\nfont-family:verdana,arial,sans-serif;\nfont-size:10pt;\nmargin:10px;\nbackground-color:#ff9900;\n}\n</style>\n</head>\n<body>\n<h3>Menu 1</h3>\n')

#for each pair of reports
for t in sort_batch:
        x = t/columns
        y = t%columns
        if x>y:
                lefthtml = htmlbase+'/'+fname_lookup[x]+'.html'
                lsource = datapath+'/'+fname_lookup[x];
                src = open(lsource,'r')
                htmldest = open(lefthtml,'w')
                htmldest.write('<html><head><title>mainframe</title></head><style>\ntextarea\n{border:1px solid #000000; width:100%; margin:3px 0; padding:3px;}</style><body><textarea>')
                htmldest.write(src.read())
                htmldest.write('</textarea></body></html>')
                htmldest.close()
                src.close()
                righthtml = htmlbase+'/'+fname_lookup[y]+'.html'
                framename = fname_lookup[x]+"_"+fname_lookup[y]+'.html'
                frame = open(htmlbase+'/'+framename,'w')
                frame.write('<html><head><title>mainframe</title></head>\n<frameset cols="50%,*" frameborder="0" border="0" framespacing="0">\n')
                frame.write('<frame name="ltab" src="'+fname_lookup[x]+'.html" marginheight="0" marginwidth="0" scrolling="auto" noresize>\n<frame name="rtab" src="'+fname_lookup[y]+'.html" marginheight="0" marginwidth="0" scrolling="auto" noresize>\n<noframes>\n<p>Nope</p>\n</noframes>\n</frameset>\n</html>\n')
                menu.write('<p><a href="'+framename+'" target="content">'+fname_lookup[x]+"_"+fname_lookup[y]+'</a>_'+batch_dist[x][y].astype('|S10')+'Marks</p>\n')
                frame.close();
                print batch_dist[x][y]
menu.write('</html>')
menu.close();

