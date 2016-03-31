#!/bin/bash

p=$(cat lastproc.log)

./keywordextract.py $p
