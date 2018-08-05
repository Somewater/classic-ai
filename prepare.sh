#!/bin/bash

BASEDIR=$(pwd)

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

cd data
wget https://dumps.wikimedia.org/ruwiki/latest/ruwiki-latest-pages-articles-multistream.xml.bz2 -O ruwiki-latest-pages-articles-multistream.xml.bz2
wget https://raw.githubusercontent.com/mhq/train_punkt/master/russian.pickle