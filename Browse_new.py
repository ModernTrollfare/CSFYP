#!/usr/bin/python

#This crawler focuses crawl on main page.

#RE handler
import re

#URLParse,urllib
import urllib

#time for pause
import time

#splinter- automated browsing to wreck webapps
from splinter import Browser

#os for system operation handling
import os

#timestamps for timed snapshots of the site
from datetime import datetime

#crawl depth constant
crawlDepth = 5;

#scroll how much
scrollCount = 1;

#token threshold: if the string is too short, we discard it
tokenThreshold = 5;

#for recording run time
start_time = time.time()

#initialise table to avoid repeated crawling
stored=["http://finance.yahoo.com/news/","http://finance.yahoo.com/"]
#stored = ["http://finance.yahoo.com/news/donald-trump-megyn-kelly-she-131126007.html","http://finance.yahoo.com/news/donald-trump-megyn-kelly-she-131126007.html"]
#initialize a RE object to make sure only "/news/" path is stored but not "(some site/news/)"
urlFilter = re.compile("/news/.*")
altFilter = re.compile("http://finance\.yahoo\.com/news/.*")

def updateCrawled(links):
        for p in links:
		url = p['href']
		#clean link from GET parameters
		url = re.sub('[\?#][^\?#]*', '',url) 

                #check if link is fresh and not with empty text(which means it would be a picture)
                if url not in stored and url not in listlog and p.text.encode('ascii','ignore')!="":
                        #debug- print the link
#                        print(p['href'])

                        #debug- print the title/link title(skipping unparsable unicodes)
#                        print(p.text.encode('ascii', 'ignore'))

			#to make sure it is not external link
			if urlFilter.match(url) != None or altFilter.match(url) != None:
        	                #store link in the table
	                        stored.append(url)
			else:
				print("not stored: "+url+'\n')
				l.write("[ %.5f ] Not Stored: %s\n" % (time.time()-start_time,url))	
                        #debug- just formatting
			
#                        print("\n");


if not os.path.exists('Data'):
    os.makedirs('Data');

os.chdir('Data')

#list of all crawled websites
if os.path.isfile('../list.log'):
        listfile = open('../list.log','r')
else:
        listfile = open('../list.log','w+')

listlog = [line.rstrip('\n') for line in listfile]

listfile.close()
listfile = open('../list.log','a')

#create folder for current time
t = datetime.now()
foldername = t.strftime('%m_%d_%Y_%H_%M_CrawlData')
os.mkdir(foldername)

#change to the new directory to put all the crawled data there
os.chdir(foldername) 

#debug: print list of links traversed
lName = 'crawlLog.log'
l = open(lName,'w');
l.write("Start Time:")
l.write(time.strftime("%d %b %Y %H:%M:%S",time.gmtime(start_time)))
l.write('\n')

#make index file
index = open('index.log','w');
ino = 1

#sets browser to PhantomJS- The default is firefox so we failed
with Browser('phantomjs') as browser:
	
	#start crawling from element 1, which is the yahoo! finance homepage
	ind = 1

	#depth is zero- the root
	currDepth = 0	

	#crawl for depth *depth defined*
	for currDepth in range (0,crawlDepth):
	
		#Breadth first search.
		#Each time, we traverse to the end of the "old list",
		#which means to crawl all the childs of nodes at depth currDepth.
		queueLimit = len(stored)
		for t in range (ind,queueLimit):
			url = stored[ind]
			print("Now processing %d of %d (depth %d): %s" % (ind,queueLimit-1,currDepth,url))
			l.write("[ %.5f ] Processing:\n%s\n" % (time.time()-start_time,url))
			browser.visit(url)
			
			#Bugs appear when pages doesn't load completely. So just try to add a pause for loading
			time.sleep(1);

			#Well, they have dead links. So this is to test where it redirected to Yahoo! homepage
			if url != browser.url and url != stored[1]:
				ind += 1
				continue;

			#if it is a "news" page, retrieve the headline and first paragraph
			if url.split('/')[-1] != '':
				sumName = ''
				firstParText = ''

				#try find the headline using css class. some pages screwed up, so i made this ugly
				newsTitle = browser.find_by_css(".headline")
				if newsTitle.is_empty():
					ind+=1
					continue
				else:
					try:
						newsTitle = newsTitle.text.encode('ascii','ignore')
					except AttributeError:
						newstitle = newsTitle.first.text.encode('ascii','ignore');
				
				#Get all the paragraphs
				Pars  = browser.find_by_css(".body.yom-art-content>p")
				#Get Creation Time- even if it is text
				cTime = browser.find_by_css("cite>abbr")
				if not cTime.is_empty():
					cTime = cTime.text.encode('ascii','ignore')
				#Same here: some pages doesnt place it in second p tag so it also became ugly- but more exceptions coming
#				if firstPar.is_empty():
#					firstPar = browser.find_by_css(".body.yom-art-content>p")
#					if firstPar.is_empty():
#						firstParText = 'Well, this page screwed up really bad- bug fix marked!'
#						sumName = 'BAD_'
#					else:
#						firstParText = firstPar.text.encode('ascii', 'ignore');
#                                else:
#	                                firstParText = firstPar.text.encode('ascii', 'ignore');
				name = url.split('/')[-1]
				#sumName = sumName + name + '.txt'
				sumName = sumName +str(ino) +'.txt'
				s = open(sumName,'w')
				l.write("Creation Time: %s\n\nTime Browsed: %s\n\n" % (cTime,time.strftime("%d %b %Y %H:%M:%S",time.gmtime(time.time()))))
				s.write(newsTitle)
				s.write('\n');
				#s.write("\n\nParagraphs:\n\n")
				for p in Pars:
					parBuffer = p.text.encode('ascii','ignore')
					lenConst = len(parBuffer.split(None))					
#					s.write('%d\n' % lenConst)
					if lenConst>tokenThreshold:
						s.write(parBuffer)
						s.write("\n\n");
				s.close()
				print 'summary saved as: '+sumName ;

			#Get the html source of current page

			#If it is not a news page, or its URL html is empty
			#if  url.split('/')[-1] == '':
			#	fName = browser.title.encode('ascii', 'ignore') + '.html'
			#else:
			#fName = url.split('/')[-1];
			fName = str(ino)+'.html'
			print 'html source saved as: '+fName
			f = open(fName, 'w')
			f.write(browser.html.encode('ascii', 'ignore'))
			f.close()

			#Scroll hard af on the main page
			if ind == 1:
				scrollVal = 10000
				for x in range (0,scrollCount):
					l.write("[ %.5f ] Scrolling\n"% (time.time() - start_time))
					browser.evaluate_script("scrollPosition={top: %d,left: 0};" % scrollVal)
					scrollVal = scrollVal + 1000
					time.sleep(5);
	
			#select links in the middle column

			#	result = browser.find_by_css(".Col2.yog-cp.Pos-a.Bxz-bb")
			#	links  = result.last.find_by_tag("a");
	
			#select links with /news/
			links = browser.find_link_by_partial_href("/news/")

			#put down the index
			index.write(str(ino)+' '+url+'\n')
			ino += 1
			
			#update crawled list. If it is the last run, just don't update the list. Wanna test how much this speeds up
			if crawlDepth != currDepth+1:
				updateCrawled(links)
			ind += 1;

l.write("Links Traversed: %d" % queueLimit)
totalTime = time.time() - start_time
l.write("End Time:")
l.write(time.strftime("%d %b %Y %H:%M:%S",time.gmtime(time.time())))
l.write('\n')
print("Processing time: %d:%d:%d" % (totalTime/3600, (totalTime%3600)/60, (totalTime%60)))
l.write("Processing time: %d:%d:%d" % (totalTime/3600, (totalTime%3600)/60, (totalTime%60)))
#l.write("List of Links:")
#for q in range(0,queueLimit):
#	l.write(stored[q])
#	l.write("\n");
for line in stored:
	if line != stored[1] and line != stored[2]:
		listfile.write(line+'\n')
listfile.close()
l.write("End of list;")
l.close();
index.close()

#log file to be processed
lastproc = open("../../lastproc.log","w");
lastproc.write(foldername);
lastproc.close();

