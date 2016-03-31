#!/bin/bash
#array=("03_25_2016_00_00_CrawlData" "03_25_2016_03_00_CrawlData" "03_25_2016_06_00_CrawlData" "03_25_2016_08_00_CrawlData" "03_25_2016_09_00_CrawlData" "03_25_2016_12_00_CrawlData" "03_21_2016_06_00_CrawlData" "03_21_2016_08_00_CrawlData" "03_21_2016_09_00_CrawlData" "03_21_2016_12_00_CrawlData" "03_21_2016_15_00_CrawlData" "03_21_2016_18_00_CrawlData" "03_21_2016_21_00_CrawlData" "03_22_2016_00_00_CrawlData" "03_22_2016_03_00_CrawlData" "03_22_2016_06_00_CrawlData" "03_22_2016_08_00_CrawlData" "03_22_2016_09_00_CrawlData" "03_22_2016_12_00_CrawlData" "03_22_2016_15_00_CrawlData" "03_22_2016_18_00_CrawlData" "03_22_2016_21_00_CrawlData" "03_23_2016_00_00_CrawlData" "03_23_2016_03_00_CrawlData" "03_23_2016_06_00_CrawlData" "03_23_2016_08_00_CrawlData" "03_23_2016_09_00_CrawlData" "03_23_2016_12_00_CrawlData" "03_23_2016_15_00_CrawlData" "03_23_2016_18_00_CrawlData" "03_23_2016_21_00_CrawlData" "03_24_2016_00_00_CrawlData" "03_24_2016_03_00_CrawlData" "03_24_2016_08_00_CrawlData" "03_24_2016_08_00_CrawlData" "03_24_2016_09_00_CrawlData" "03_24_2016_12_00_CrawlData" "03_24_2016_15_00_CrawlData" "03_24_2016_18_00_CrawlData" "03_24_2016_21_00_CrawlData")

array=($(ls experimental))

for p in "${array[@]}"
do
#echo $p
mkdir "experimental/$p"
mkdir "experimental/$p/summary"

./summextract.py $p

done

