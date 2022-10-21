# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
TREC question-type classification
'''

from __future__ import absolute_import, division, unicode_literals

import os
import io
import logging
import numpy as np
import random
from senteval.tools.validation import KFoldClassifier


class CLSEval(object):
    def __init__(self, task_path, seed=1111, zero_shot=False):
        logging.info('***** Transfer task : CLS (zero-shot = {}) *****\n\n'.format(zero_shot))
        self.seed = seed
        self.zero_shot = zero_shot
        
        self.langs = ['de','en','fr','ja']
        if zero_shot:
            self.train = {'en': self.loadFile(os.path.join(task_path, 'cls.train.raw.en.tsv'))}
        else:
            self.train = {lang: self.loadFile(os.path.join(task_path, 'cls.train.raw.{}.tsv'.format(lang))) for lang in self.langs}
        self.test = {lang: self.loadFile(os.path.join(task_path, 'cls.test.raw.{}.tsv'.format(lang))) for lang in self.langs}

    def do_prepare(self, params, prepare):
        #samples = self.train['X'] + self.test['X']
        #return prepare(params, samples)
        return

    def loadFile(self, fpath):
        trec_data = {'X': [], 'y': []}
        tgt2idx = {'0': 0, '1': 1}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                target, sample = line.strip().split('\t', 1)
                assert target in tgt2idx, target
                trec_data['X'].append(sample)
                trec_data['y'].append(tgt2idx[target])
        return trec_data

    def run(self, params, batcher):
        # Sort to reduce padding
        sorted_corpus_train = {k: sorted(zip(v['X'], v['y']),
                                    key=lambda z: (len(z[0]), z[1])) for k, v in self.train.items()}
        train_samples = {k: [x for (x, y) in v] for k, v in sorted_corpus_train.items()}
        train_labels_ = {k: [y for (x, y) in v] for k, v in sorted_corpus_train.items()}

        sorted_corpus_test = {k: sorted(zip(v['X'], v['y']),
                                    key=lambda z: (len(z[0]), z[1])) for k, v in self.test.items()}
        test_samples = {k: [x for (x, y) in v] for k, v in sorted_corpus_test.items()}
        test_labels = {k: [y for (x, y) in v] for k, v in sorted_corpus_test.items()}

        # Get train embeddings
        train_embeddings = []
        train_labels = []
        for key in train_samples:
            for ii in range(0, len(train_labels_[key]), params.batch_size):
                batch = train_samples[key][ii:ii + params.batch_size]
                embeddings = batcher(params, batch, lang=key[-2:])
                train_embeddings.append(embeddings)
                train_labels.extend(train_labels_[key][ii:ii + params.batch_size])
            logging.info('Computed train embeddings for {}'.format(key))
        train_embeddings = np.vstack(train_embeddings)

        # Get test embeddings
        test_embeddings = {}
        for key in test_samples:
            test_embeddings_ = []
            for ii in range(0, len(test_labels[key]), params.batch_size):
                batch = test_samples[key][ii:ii + params.batch_size]
                embeddings = batcher(params, batch, lang='en')
                test_embeddings_.append(embeddings)
            test_embeddings[key] = np.vstack(test_embeddings_)
            logging.info('Computed test embeddings for {}'.format(key))

        config_classifier = {'nclasses': 2, 'seed': self.seed,
                             'usepytorch': params.usepytorch,
                             'classifier': params.classifier,
                             'kfold': params.kfold}
        clf = KFoldClassifier({'X': train_embeddings,
                               'y': np.array(train_labels)},
                              {'X': test_embeddings,
                               'y': {k: np.array(v) for k, v in test_labels.items()}},
                              config_classifier)
        devacc, testacc, _ = clf.run()
        logging.debug(('\nDev acc : {0} Test acc : {1} ' 
            'for CLS (zero-shot = {2})\n').format(devacc, ', '.join(['({}) {}'.format(k, v) for k, v in testacc.items()]), self.zero_shot))
        return {'devacc': devacc, 'acc': testacc}
                #'ndev': len(self.train['X']), 'ntest': len(self.test['X'])}
