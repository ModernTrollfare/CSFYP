#! /usr/bin/env python

#An implementation of "Automatic Text Summarization using a Machine Learning
#Approach", Neto,J. , Freitas,A. , Kaestner, C.
#Preprossing script.

#Input: A single document
#Output: Preprocessed Document for vector generation

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

#Parameters
#dumping preprocess file
prep_dbg = False
#showing process
dbg_verbose = True 
#Dumping the dendrogram
printgraph = True
#graph path
graphpic = 'foo.png'
if printgraph:
	import matplotlib
	#Dumping from remote/No display
	matplotlib.use('Agg')
	from matplotlib import pyplot as plt

#Anaphor List
ana_list = ["his","her","this","that","those","these","their","our"]

#Discourse words list
"""
lists of single words signaling discourse of eleboration.
Taken from
http://russell.famaf.unc.edu.ar/~laura/shallowdisc4summ/discmar/#vague_dms
This list is "aggressive" i.e. may have high level of FPs.
"""
disc_list = ["despite","although","except","because","due","given","specifically","essentially","comparison","particular","particularly","example","precisely","considering","after","before","originally","during","while","unless","when","where","accordance","between","towards","until","following"]


#"""
#Calculates the cosine distance between sentence and query.
#Implementation of
#Salton, G. et al, Term-weighting Approaches in automatic text retrieval.
#"""

#build tfisf dictionary
def idf_build(tokenlist,df):
	idf = defaultdict(lambda:0)
	for t in set(tokenlist):
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
		mat[ind] = [float(x)/slen for x in mat[ind]]
	tf = np.multiply(np.array(mat),idf)
	return tf

#print debug for dictionary
def tfisf_dbg(tdict):
	for key in tdict:
		print "%s\t%f" % (key,tdict[key])
	exit()
	return

#Calculates TFISF average value for one sentence.
def calc_tfisf(tdict,sent):
	tfisf_rate = 0;
	tokfreq = len(sent)
	for token in sent:
		tfisf_rate = tfisf_rate + (1.0/tdict[token])
	return tfisf_rate/tokfreq

def calc_dist(sent,query,tdict):
	#calculate sum(w_sk^2)
	sentsum = 0;
	for token in set(sent):
		sentsum = sentsum + (1.0/(tdict[token]*tdict[token]))
	#calculate sum(w_qk^2)
	querysum = 0;
	for token in set(query):	
		querysum = querysum + (1.0/(tdict[token]*tdict[token]))
	#sqrt(sum(w_sk^2)*sum(w_qk^2)
	denom = math.sqrt(sentsum*querysum)
	#calculate sum(w_qk*w_sk)
	numerator = 0 
	for key in tdict:
		if ((key in set(sent)) & (key in set(query))):
			numerator = numerator + (1.0/(tdict[key]*tdict[key]))
	try:
		return numerator/denom
        except ZeroDivisionError:
                print set(query)
                print set(sent)
                exit()

#Calculate the centroid of all sentences
def calc_centroid(data,tdict):
	centdict = defaultdict(lambda:0);
	for sent in data:
		sentset = set(sent)
		for elem in sentset:
			centdict[elem] += 1.0/tdict[elem]
	centdict = dict((key,item/len(data)) for (key,item) in centdict.iteritems())
	return centdict

#Calculate sentence-wise centroid distance
def calc_centdist(sent,centroid,tdict):
        #calculate sum(w_sk^2)
        sentsum = 0;
        for token in set(sent):
                sentsum = sentsum + (1.0/(tdict[token]*tdict[token]))
        #calculate sum(w_ck^2)
        centroidsum = sum([c*c for c in centroid.values()])
	denom = math.sqrt(centroidsum*sentsum)
	#calculate sum(w_ck*w_sk)
	numerator = 0
	for token in set(sent):
		numerator = numerator + (centroid[token]/tdict[token])
	return numerator/denom	

#Construct most frequent noun list/dict
def calc_nounlist(sentlist):
	if dbg_verbose:
		print "Generating noun list....."
	noundict = defaultdict(lambda:0)
	for sent in sentlist:
		sent_pos = nltk.word_tokenize(sent)
		if dbg_verbose:
			print "tagging POS: %s" % sent
        	pos_tuple = nltk.pos_tag(sent_pos)
		for x,y in pos_tuple:
			x = x.lower()
			if y in ['NN','NNP']:
				noundict[x] += 1
	noundict = sorted(noundict.items(), key=operator.itemgetter(1))
	if dbg_verbose:
		print "Done."
	if len(noundict) >= 15:
		return [x for x,y in noundict[0:15]]
	else:
		return [x for x,y in noundict]

#Check if element in list
def inList(listdict,sentlist):
	for t in sentlist:
		if t.lower() in listdict:
			return True
	return False

#Constructing Hierachical Agglomerative Clustering Tree
def buildHAC(tf_matrix):
	if dbg_verbose:
		print "Constructing HAC Tree...."
	"""
	#construct TF-ISF vector indices
	dvecind = [x for x,y in tdict.items()]
	#m*n matrix list
	hacvec = [None]*len(los)
	#construct for each sentence
	for raw,sent,ind in los:
		vec = [0.0] *len(dvecind)
		for t in sent:
			vec[dvecind.index(t)] = 1.000/tdict[t]
	#		print t,tdict[t]
		hacvec[ind] = vec
		if printgraph:
			print ind,raw
	#put everything into a numpy matrix
	#Explicit cast to float32 in case we throw it to GPU
	hacdistmatrix = np.array(hacvec,dtype=np.float32)
	"""
	#get pdist distance matrix
	hacdist = scipy.spatial.distance.pdist(tf_matrix,'cosine')
	if dbg_verbose:
		print "Done."
	#UPGMA Clustering - get tree
	Z = scipy.cluster.hierarchy.average(hacdist)
	return scipy.cluster.hierarchy.to_tree(Z),Z

#Processing the HAC tree to get depth and location.
def proc_HACtree(htree,cdepth=0,steplist=[0.0,0.0,0.0,0.0]):
	if htree.is_leaf():
		return [(htree.get_id(),float(cdepth),steplist)]
	if cdepth < 4:
		leftlist = copy.deepcopy(steplist)
		leftlist[cdepth] = -1.0
		rightlist = copy.deepcopy(steplist)
		rightlist[cdepth] = 1.0
		return proc_HACtree(htree.get_left(),cdepth+1,leftlist)+proc_HACtree(htree.get_right(),cdepth+1,rightlist)
	else:
		return proc_HACtree(htree.get_left(),cdepth+1,steplist)+proc_HACtree(htree.get_right(),cdepth+1,steplist)

#load stopwords list
stopwords = nltk.corpus.stopwords.words('english')

#load up the sentence parser
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

#Define Regex cleaner for step 3.
regex = re.compile('[\=(),\.!\?\;"\']')

#Define Stemmer
stemmer = nltk.PorterStemmer()

#input file path in arguments
path = os.sys.argv[1]

#open file
doc = open(path,'r')

#read document
#title line
title = ['\n']
while (title == ['\n']):
	title = doc.readline()
#rest of document
data = doc.read()

#Step 1: separate sentences
sentlist = tokenizer.tokenize(data)
#put title to head
sentlist = [title] + sentlist

#construct noundict
noundict = calc_nounlist(sentlist)

#Step 2: Case Folding
sentlist = [t.lower() for t in sentlist]

#processed outfile.
if prep_dbg:
	processed = open("prep_"+path,'w')

#list of sentences and preprocessed form
los=[None]*(len(sentlist)-1);

#title tuple
ptitle = None

#entry index
index = -1;

#Steps 3 and 4
for sent in sentlist:
	nlist = sent.split()
	#remove all special characters in the sentence.
	nlist = [regex.sub('', token) for token in nlist]
	#Step 3: Remove stopwords
	nlist = [t for t in nlist if t not in stopwords]
	#Step 4: Stemming (I.E. Removing suffixes)
	nlist = [stemmer.stem(t) for t in nlist]
	#Step 5: Print/Write the result.
	if (index == -1):
		ptitle = (title,nlist)
	else:
		los[index] = (sent,nlist,index);
	#Dump to preprocessed dump
	if prep_dbg:
		for t in nlist:
			processed.write(t)
			processed.write(" ")
		#If all words are removed
		if not nlist:
			processed.write("notokenshere")
		processed.write("\n")
	#Increment index
	index += 1
#Close dump
if prep_dbg:
	processed.close()
doc.close();

###################
#Vector Generation#
###################

#preprocessed title
nul,prep_title = ptitle;

#rest of preprocessed data
prep = []
for x,y,z in los:
	prep += y
if dbg_verbose:
	print "Getting Keywords from MAUI......"
#get MAUI Keywords.
DEVNULL = open(os.devnull, 'wb')
#return exit code
krecode = subprocess.check_call(["./keywdgen.sh", path], stdout=DEVNULL, stderr=subprocess.STDOUT)
DEVNULL.close()
#read keywords into a list.
kwdfile = open("in.key",'r')
kwdlist = kwdfile.read().rsplit()
kwdfile.close()
if dbg_verbose:
	print "Done."
#stem the results
finkwdlist = [regex.sub('', token.lower()) for token in kwdlist]
finkwdlist = [stemmer.stem(t) for t in finkwdlist]

#line by line
prepdata = [y for x,y,z in los]

#whole preprocessed document for tfisf lookup construction
toklist = prep+prep_title
isf,tfisf_dict = idf_build(toklist,len(los))

#Consruct the tf part;
#gets the matrix out
tf = tf_build(los,isf,tfisf_dict);

#max sentence length
max_len = float(max([len(sent) for sent in prepdata]))

#Data Centroid
centvector = tf.mean(0);

#pairwise distance of sentence
sentdist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(tf,'cosine'))

#construct keyword tfidf vector
kw_tf = tf_build([(title,finkwdlist,0)],isf,tfisf_dict);
kw_lookup_tf = np.concatenate((tf,kw_tf));
kw_dist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(kw_lookup_tf,'cosine'))

#construct title tfidf vector
title_tf = tf_build([(title,prep_title,0)],isf,tfisf_dict)
whole_tf = np.concatenate((tf,title_tf));
alldist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(whole_tf,'cosine'));

#construct the distance to centroid
cv = np.reshape(centvector,(1,-1))
cv_tf = np.concatenate((tf,cv));
cv_dist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(cv_tf,'cosine'))

prepdata = [s for s in prepdata if not (s=='')]

#holding all vectors
rvector = [None]*len(los)

#build HAC tree
(htree,Z) = buildHAC(tf)

#Process HAC tree
htreeres = proc_HACtree(htree=htree)

for raw,sent,ind in los:
	vec = [None]*16
	#print sent
	#Process Sentences with no meaningful words
	if (len(sent) ==0):
		vec[0] = 0.0;
		vec[1] = 0.0;
		vec[2] = float(ind)/(len(los)-1);
		vec[3:16] = 0.0;
		rvector[ind] = vec
		continue
	#Term 1: mean-TF-ISF
	vec[0] = tf[ind].mean()
	#Term 2: Sentence Length
	vec[1] = len(sent)/max_len
	#Term 3: Sentence Position
	vec[2] = float(prepdata.index(sent))/(len(prepdata)-1)
	#Term 4: Similarity to title
#	vec[3] = calc_dist(sent,prep_title,tfisf_dict)
	vec[3] = alldist[ind,len(los)]
	#Term 5: Similarity to keywords
	#Obtain Keywords from MAUI.
	"""
	Keywords extracted using MAUI-1.2.
	Model trained using news sources and keyword sources from:
	Luis Marujo and Anatole Gershman and Jaime Carbonell and Robert Frederking and Joao P. Neto,
	Supervised Topical Key Phrase Extraction of News Stories using Crowdsourcing,
	Light Filtering and Co-reference Normalization.
	"""
#	vec[4] = calc_dist(sent,finkwdlist,tfisf_dict)
	vec[4] = kw_dist[ind,len(los)]
	#vec[4] = 0;
	#Term 6: Sentence to Sentence Cohesion
	"""
	vec[5] = 0
	for x,y,z in los:
		if ((not (y == sent)) or (len(y) == 0)):
			vec[5] =vec[5] + calc_dist(sent,y,tfisf_dict)	
	"""
	vec[5] = sentdist[ind].sum()
	#Term 7: Sentence to Centroid Cohesion
	vec[6] = cv_dist[ind,len(los)];
	#Retrieve Hierachical Agglomerative Clustering Tree
	#Term 8: Node Depth
	vec[7] = 0.0;
	#Term 9: Node Position (from path)
	vec[8] = 0.0
	vec[9] = 0.0
	vec[10] = 0.0
	vec[11]	= 0.0
	#Semantic Structure Attributes
	raw_pos = nltk.word_tokenize(raw)
	rawlist =  nltk.pos_tag(raw_pos);
	raw_nounlist = [x for x,y in rawlist if y in ["NN","NNP"]]
	#Term 10: Indicator of Main Concepts
	if inList(noundict,raw_nounlist):
		vec[12] = 1.00
	else:
		vec[12] = 0.00
	#Term 11: Anaphor Removal
	if inList(ana_list,raw_pos[0:6]):
		vec[13] = 1.0
	else:
		vec[13] = 0.00
	#Term 12: Proper Nouns
	nnplist = [x for x,y in rawlist if y=="NNP"]
	if nnplist:
		vec[14] = 1.00
	else:
		vec[14] = 0.00
	#Term 13: Discourse Markers
	if inList(disc_list,raw_pos):
		vec[15] = 1.00
	else:
		vec[15] = 0.00
	rvector[ind] = vec

#Extracting the results from processed tree to the vectors
for x,y,z in htreeres:
	rvector[x][7] = y;
	rvector[x][8:12] = z[0:5]

print "Pre-Normalized Values:"
#Dump vectors to stdout.
for vec in rvector:
	print vec

#Normalize data.
for i in range(0,11):
	maxcolval = max([l[i] for l in rvector])
	mincolval = min([l[i] for l in rvector])	
	for ent in rvector:
		if i in range(8,4):
			ent[i] = (ent[i]+1.0)/2.0
		else:
			ent[i] = (ent[i]-mincolval)/(maxcolval-mincolval)

print "Normalized Values"
#Dump vectors to stdout.
for vec in rvector:
	print vec

#Dumping the dendrogram.
if printgraph:
	plt.figure(figsize=(25, 10))
	plt.title('Hierarchical Clustering Dendrogram')
	plt.xlabel('sample index')
	plt.ylabel('distance')
	scipy.cluster.hierarchy.dendrogram(
	    Z,
	    leaf_rotation=90.,  # rotates the x axis labels
	    leaf_font_size=8.,  # font size for the x axis labels
	)
	plt.savefig(graphpic)
