#!/bin/bash

p=$(cat lastproc.log)

mkdir "experimental/$p"

./newinterclus.py $p

./newtimeclus.py
