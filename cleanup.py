#!/usr/bin/python

import os

import sys

import glob

os.chdir('Data')

currDir = os.getcwd()

filelist = os.listdir(currDir)

#print filelist

stored = []

for f in filelist:
	if os.path.isdir(f):
		os.chdir(f)
#		l = open('delList.log','w')
		for x in glob.glob("*.html"):
			string = "https://finance.yahoo.com/news/"+x
			string = string.strip('\n')
			if string not in stored:
				stored.append(string)
			else:
				os.remove(x)
#		l.close()
		os.chdir("..")
#for line in stored:
#	print line

