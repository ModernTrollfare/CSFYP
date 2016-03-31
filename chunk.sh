#!/bin/bash

#files="Data/*"

files="*"

cd Data/

for p in $files
do
#echo $f

shopt -s nullglob
for f in "/Data/$p/*.txt"
do
#fil='/home/toor/Desktop/CSFYP/NewsDigest/Data/09_08_2015_15_00_CrawlData/china-stocks-fall-more-1-013756463.html.txt' 

#echo "$f"

fname="${f##*/}"

#echo "CrawlData/$p/"$fname'.log'

/usr/bin/python chunk_proto.py "$f" >"openie_jar/CrawlData/$p/keywords_"$fname'.log' 2>&1
/usr/bin/python chunk_regex.py "$f" >"openie_jar/CrawlData/$p/re_keywords_"$fname'.log' 2>&1
done
done

