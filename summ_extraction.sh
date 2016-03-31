#!/bin/bash

p=$(cat lastproc.log)

mkdir "experimental/$p"
mkdir "experimental/$p/summary"

./summextract.py $p
