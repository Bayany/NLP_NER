#!/usr/bin/env python

import logging
import sys
targets = logging.StreamHandler(sys.stdout), logging.FileHandler('run.log')
logging.basicConfig(format='%(message)s', level=logging.INFO, handlers=targets)


import os


for d in ["experitments","models","reports","stats","logs"]:
    if(not os.path.exists(d)):
        os.mkdir(d)

# run word2vec
logging.info("Traning word2vec:")
os.system("python src/word2vec/run.py")
logging.info("Saved word2vec vectors in /models./nGenerating report:")
os.system("python src/word2vec/load_model.py")
logging.info("Saved report in /reports.")

# run language_model
logging.info("Traning language model:")
os.system("python src/language_model.py")
logging.info("Saved word2vec vectors in /models.")