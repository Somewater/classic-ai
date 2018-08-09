import argparse
import my
from my import *
from collections import Counter
from gensim.models import Word2Vec
import logging

arg_parser = argparse.ArgumentParser(description="Script readme")
arg_parser.add_argument('--window', default='5')
arg_parser.add_argument('--negative', default='5')
arg_parser.add_argument('--min_count', default='5')
arg_parser.add_argument('--ns_exponent', default='0.75')
arg_parser.add_argument('--cbow_mean', default='1')
arg_parser.add_argument('--epochs', default='5')
arg_parser.add_argument('--alpha', default='0.025')
arg_parser.add_argument('--alpha_dec', default='0.001')
arg_parser.add_argument('--sample', default='0.001')
arg_parser.add_argument('--sg', default='0')
arg_parser.add_argument('--hs', default='0')
arg_parser.add_argument('--log', default='measure_w2v_acc.log')
args = arg_parser.parse_args()

window = int(args.window)
negative = int(args.negative)
min_count = int(args.min_count)
ns_exponent = float(args.ns_exponent)
cbow_mean = int(args.cbow_mean)
epochs = int(args.epochs)
alpha = float(args.alpha)
alpha_dec = float(args.alpha_dec)
sample = float(args.sample)
sg = int(args.sg)
hs = int(args.hs)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO, filename=args.log)
reader = DataReader()
corpus = WikiCorpus(reader, type='lemm')
corpusw2v = CorpusW2v(corpus, reader)

corpusw2v.model.window = window
corpusw2v.model.negative = negative
corpusw2v.model.ns_exponent = ns_exponent
corpusw2v.model.cbow_mean = cbow_mean
corpusw2v.model.epochs = 1
corpusw2v.model.sample = sample
corpusw2v.model.sg = sg
corpusw2v.model.hs = hs
corpusw2v.model.min_count = min_count

current_alpha = alpha
max_accuracy = -1
for epoch in range(epochs):
    corpusw2v.model.alpha = current_alpha
    logging.info("Epoch %d on %s" % (epoch, repr(corpusw2v.model)))
    corpusw2v.train()
    acc, acc_info = corpusw2v.accuracy()
    logging.info("Epoch %d result = %f\n%s" % (epoch, acc, "\n".join(['\t' + l for l in repr(acc_info).splitlines()])))
    if acc > max_accuracy:
        max_accuracy = acc
    else:
        logging.warning("Accuracy degradation %f->%f on %d epoch" % (max_accuracy, acc, epoch))
    current_alpha -= alpha_dec
    if current_alpha <= 0:
        logging.error('alpha is not positive, stop on %d epoch' % epoch)
        break
print(max_accuracy)
