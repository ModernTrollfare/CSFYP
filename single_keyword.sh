#!/bin/bash

array=($(ls experimental))

for p in "${array[@]}"
do

./keywordextract.py $p

done
