#!/bin/bash

BASEDIR=$(pwd)
ln -s $BASEDIR/data $BASEDIR/my/data

conda create -n sber2
source activate sber2

python -c '
import os
with open("images/sberbank-python/pip_freeze.txt") as f:
    for l in f:
        if "==" in l:
            print(l)
            os.system("pip install %s" % l)
'

wget https://dumps.wikimedia.org/ruwiki/latest/ruwiki-latest-pages-articles-multistream.xml.bz2 -O data/ruwiki-latest-pages-articles-multistream.xml.bz2