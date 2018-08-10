import argparse

from gensim.models.callbacks import CallbackAny2Vec

import my
from my import *
from collections import Counter
from gensim.models import Word2Vec
import logging

arg_parser = argparse.ArgumentParser(description="Script readme")
arg_parser.add_argument('--size', default='100')
arg_parser.add_argument('--window', default='5')
arg_parser.add_argument('--negative', default='5')
arg_parser.add_argument('--min_count', default='5')
arg_parser.add_argument('--ns_exponent', default='0.75')
arg_parser.add_argument('--cbow_mean', default='1')
arg_parser.add_argument('--epochs', default='5')
arg_parser.add_argument('--alpha', default='0.025')
arg_parser.add_argument('--min_alpha', default='0.0001')
arg_parser.add_argument('--sample', default='0.001')
arg_parser.add_argument('--sg', default='0')
arg_parser.add_argument('--hs', default='0')
arg_parser.add_argument('--log', default='measure_w2v_acc.log')
args = arg_parser.parse_args()

size = int(args.size)
window = int(args.window)
negative = int(args.negative)
min_count = int(args.min_count)
ns_exponent = float(args.ns_exponent)
cbow_mean = int(args.cbow_mean)
epochs = int(args.epochs)
alpha = float(args.alpha)
min_alpha = float(args.min_alpha)
sample = float(args.sample)
sg = int(args.sg)
hs = int(args.hs)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO, filename=args.log)
reader = DataReader()
corpus = WikiCorpus(reader, type='lemm')
corpusw2v = CorpusW2v(corpus, reader, vector_size=size)

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

callback = CallbackAny2Vec()
if corpusw2v.model.callbacks:
    corpusw2v.model.callbacks.append()
else:
    corpusw2v.model.callbacks = [callback]


corpusw2v.train(alpha, min_alpha, epochs)


for epoch in range():
    logging.info("Epoch %d on %s" % (epoch, repr(corpusw2v.model)))

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

import sys
sys.exit()

# draft

import hyperopt
from hyperopt import hp, fmin
import subprocess
import math
import traceback

def measure_w2v_acc_wrapper0(size, window, negative, min_count, epochs, alpha, min_alpha=0.0001, ns_exponent = 0.75, cbow_mean = 1, sample = 0.001, sg = 0, hs = 0):
    params_str = "size=%d, window=%d, negative=%d, min_count=%d, epochs=%d, alpha=%f" % (size, window, negative, min_count, epochs, alpha)
    print("Exec with %s" % params_str)
    # return 1 - math.fabs(alpha - 0.02)
    try:
        proc = subprocess.Popen([str(v) for v in [
            'python',
            'measure_w2v_acc.py',
            '--size', int(size),
            '--window', int(window),
            '--negative', int(negative),
            '--min_count', int(min_count),
            '--ns_exponent', ns_exponent,
            '--cbow_mean', int(cbow_mean),
            '--epochs', int(epochs),
            '--alpha', alpha,
            '--min_alpha', min_alpha,
            '--sample', sample,
            '--sg', int(sg),
            '--hs', int(hs)]], stdout=subprocess.PIPE)
        proc_out, proc_err = proc.communicate()
        score = float(proc_out.splitlines()[-1])
        print("Result=%f with %s" % (score, params_str))
        return -score # because big score is good
    except Exception as e:
        print("ERROR: %s with %s" % (repr(e), params_str))
        traceback.print_exc()
        return 1000000


def measure_w2v_acc_wrapper(args):
    return measure_w2v_acc_wrapper0(*args)

space = [
    hp.choice('size', [100, 200, 500, 1000]),
    hp.quniform('window', 5, 1000, 1),
    hp.quniform('negative', 5, 100, 1),
    hp.quniform('min_count', 5, 1000, 1),
    hp.choice('epochs', [1, 5, 20, 50]),
    hp.uniform('alpha', 0.01, 0.5)
]
best_params = fmin(fn=measure_w2v_acc_wrapper, space=space, algo=hyperopt.tpe.suggest, max_evals=100)
print(best_params)