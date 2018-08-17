import argparse

from gensim.models.callbacks import CallbackAny2Vec

import my
from my import *
from collections import Counter
from gensim.models import Word2Vec
import logging
import time
import warnings

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
arg_parser.add_argument('--save', default='')
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
#if not os.path.isfile(reader.get_tmp_filepath('mode_with_vocab.bin')):
#    corpusw2v.model.build_vocab(corpusw2v.sentences())
#    corpusw2v.model.save(reader.get_tmp_filepath('mode_with_vocab.bin'))

class MyCallback(CallbackAny2Vec):
    def __init__(self, reporter, param, reporter_fileobj):
        self.reporter = reporter
        self.param = param
        self.reporter_fileobj = reporter_fileobj

    def on_train_begin(self, model):
        self.epoch_number = 1
        self.start_time = time.time()
        self.max_accuracy = -1
        self.max_accuracy_epoch = 0
        self.full_duration_secods = None

    def on_epoch_begin(self, model):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, model):
        epoch_duration_seconds = time.time() - self.epoch_start_time
        acc, acc_info = corpusw2v.accuracy()
        logging.info("Epoch %d result = %f after %.1f seconds\n\t%s"
                     % (self.epoch_number, acc, epoch_duration_seconds, repr(acc_info)))
        if acc > self.max_accuracy:
            self.max_accuracy = acc
            self.max_accuracy_epoch = self.epoch_number
        elif acc < self.max_accuracy:
            logging.warning("Accuracy degradation %f->%f on %d epoch" % (self.max_accuracy, acc, epochs))
        print('%d/%d\t%.10f' % (epochs, self.epoch_number, acc))

        analogy_acc = corpusw2v.analogy_accuracy()
        accuracy = self.max_accuracy
        row = [self.epoch_number, self.param['size'], self.param['window'], self.param['negative'],
               self.param['min_count'], self.param['alpha'], self.param['sample'], self.param['sg'], self.param['hs'],
               accuracy, analogy_acc]
        self.reporter.writerow([str(i) for i in row])
        self.reporter_fileobj.flush()

        self.epoch_number += 1

    def on_train_end(self, model):
        self.full_duration_secods = time.time() - self.start_time

report_inited = not os.path.isfile('report.csv')
reporter_fileobj = open('report.csv', 'a')
reporter = csv.writer(reporter_fileobj)
if report_inited:
    reporter.writerow(['epoch', 'size', 'window', 'negative', 'min_count', 'alpha', 'sample', 'sg', 'hs', 'accuracy', 'analogy_acc'])
params = []
for window in [2, 5, 20, 50, 200, 500]:
    for negative in [2 , 5, 50]:
        for min_count in [5, 20, 100]:
            for alpha in [0.0005, 0.001, 0.002, 0.005]:
                for sample in [0.0001, 0.001]:
                    for sg in [0, 1]:
                        for hs in [0, 1]:
                            params.append({'size': size, 'window': window, 'negative': negative,
                                           'min_count': min_count, 'alpha': alpha, 'sample': sample,
                                           'sg': sg, 'hs': hs})


for param_idx, param in enumerate(params):
    print("Param %d from  %d: %s" % (param_idx, len(params), repr(param)))
    window = param['window']
    negative = param['negative']
    min_count = param['min_count']
    alpha = param['alpha']
    sample = param['sample']
    sg = param['sg']
    hs = param['hs']

    callback = MyCallback(reporter, param, reporter_fileobj)
    #corpusw2v.model = Word2Vec.load(reader.get_tmp_filepath('mode_with_vocab.bin'))
    corpusw2v = CorpusW2v(corpus, reader, vector_size=size)
    corpusw2v.model.window = window
    corpusw2v.model.negative = negative
    corpusw2v.model.ns_exponent = ns_exponent
    corpusw2v.model.cbow_mean = cbow_mean
    corpusw2v.model.epochs = 3
    corpusw2v.model.sample = sample
    corpusw2v.model.sg = sg
    corpusw2v.model.hs = hs
    corpusw2v.model.min_count = min_count

    corpusw2v.model.callbacks = [callback]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corpusw2v.train(alpha, min_alpha, epochs)
    logging.info("Best accuracy is %f (%d epoch from %d) after %.1f seconds" %
                 (callback.max_accuracy, callback.max_accuracy_epoch, epochs, callback.full_duration_secods))
    if args.save:
        corpusw2v.model.wv.save_word2vec_format(args.save)
    analogy_acc = corpusw2v.analogy_accuracy()
    accuracy = callback.max_accuracy
    print('Analogy accuracy: %.10f' % analogy_acc)
    print('Accuracy: %.10f' % accuracy)

    row = [1, size, window, negative, min_count, alpha, sample, sg, hs, accuracy, analogy_acc]
    reporter.writerow([str(i) for i in row])

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