#!/usr/bin/python

#nltk nlp python toolkit
import nltk
#

#operator
import operator

#string
import string

#Chunker Class
class ChunkParser(nltk.ChunkParserI):
	def __init__(self, train_sents):
		train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)] for sent in train_sents]
		self.tagger = nltk.TrigramTagger(train_data)
	
	def parse(self,sentence):
		pos_tags = [pos for (word,pos) in sentence]
		tagged_pos_tags = self.tagger.tag(pos_tags)
		chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
		conlltags = [(word,pos,chunktag) for ((word,pos),chunktag) in zip(sentence,chunktags)]
		return nltk.chunk.conlltags2tree(conlltags)


#storing (phrase,count) pairs
nounphrases = dict()

#Igonre List
IGN_LIST = ['-------end','of','all','that','this','those','who','which','these','for','when','where','extraction-------','a', 'an', 'and','the' ,'at_time', 'he', 'she', 'it' ,'they' ,'their','these','our','to','we','is', 'am', 'are','be','was','were']

#define extraction type: 0 for trained model; 1 for regex parsing
exttype =0 

def dupdate(obj):
	if obj in nounphrases:
		nounphrases[obj] += 1
	else:
		nounphrases[obj] = 1



#Tree traversal/extraction
def extraction(tree):
	try:
		tree.label()
	except AttributeError:
		return
	else:
		if tree.label() == 'NP': #If it is a noun phrase
			tmplist = [x for (x,y) in tree.leaves()]
			if len(tmplist)<3:
				tmpstr = ' '.join(tmplist)
		#		print tmpstr
				dupdate(tmpstr)
		else:
			for c in tree:
				extraction(c)
#argument parser
import argparse

parser = argparse.ArgumentParser(description='Processing the outputs given by ReVerb')
parser.add_argument('filepath', metavar='path/to/file' , nargs = 1)

args = parser.parse_args()

#get the filepath of the file
fName = args.filepath[-1]

f = open(fName).read()

#POS Tagging, etc
strings = nltk.sent_tokenize(f)
strings = [nltk.word_tokenize(s.translate(None,string.punctuation)) for s in strings]
strings = [nltk.pos_tag(s) for s in strings]


if exttype == 0:
	#train up a chunker
	testbase = nltk.corpus.conll2000.chunked_sents('test.txt',chunk_types=['NP'])
	trainbase = nltk.corpus.conll2000.chunked_sents('train.txt', chunk_types=['NP'])

	Chunker = ChunkParser(trainbase)

else:
	#define the regex
	grammar = r"""
		NP: {<NN>+}		
		    {<NNP>+}		
		    {<JJ>*<NN>}		
	"""

#                NP: {<NN>+}             #consecutive nouns
#                    {<NNP>+}            #sequence of proper nouns
#                    {<JJ>*<NN>}         #adjectives and noun

	#Define the chunker
	Chunker = nltk.RegexpParser(grammar)
	
#extract: tree recursion
for s in strings:
        t = Chunker.parse(s);
        #print(t);
        extraction(t);

#sort dict by occurrence

sortednouns = sorted(nounphrases.items(), key=operator.itemgetter(1), reverse=True)

#print (key,value) pairs
for key in sortednouns:
#	ttuple = (key,sortednouns[key])
#	print ttuple
	(x,y) = key
	if x.lower() not in IGN_LIST:
		print key

